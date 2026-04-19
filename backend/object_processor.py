import cv2
import base64
import io
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from ultralytics import YOLO
import db

CONFIDENCE_THRESHOLD = 0.4  # Lowered to 0.4 to pick up smaller objects
PERSON_CLASS_ID = 0  # COCO class 0 = person
NMS_IOU_THRESHOLD = 0.5  # IoU threshold for deduplicating YOLO detections
MATCH_IOU_THRESHOLD = 0.3  # IoU threshold for deduplicating matched objects


def _compute_iou(box_a, box_b):
    """Compute Intersection over Union between two boxes {x, y, w, h} in pixels."""
    ax1, ay1 = box_a["x"], box_a["y"]
    ax2, ay2 = ax1 + box_a["w"], ay1 + box_a["h"]
    bx1, by1 = box_b["x"], box_b["y"]
    bx2, by2 = bx1 + box_b["w"], by1 + box_b["h"]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = box_a["w"] * box_a["h"]
    area_b = box_b["w"] * box_b["h"]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_detections(detections, iou_threshold):
    """Non-maximum suppression: remove overlapping detections, keep higher confidence."""
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        sorted_dets = [
            d for d in sorted_dets
            if _compute_iou(best, d) < iou_threshold
        ]

    return keep


class ObjectProcessor:
    def __init__(self):
        self.yolo_model = YOLO("yolo26s.pt")

        # Local DINOv2 model for geometric instance embeddings (384-dim)
        # Use MPS on Apple Silicon for fast inference
        if torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        print(f"[ObjectProcessor] Loading Meta DINOv2 (facebook/dinov2-small) on {self._device}...")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-small').to(self._device)
        self.dino_model.eval()
        print("[ObjectProcessor] DINOv2 model loaded successfully.")

    def _decode_frame(self, frame_b64: str):
        img_data = base64.b64decode(frame_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _run_yolo(self, img, conf_threshold=None):
        """Run YOLO and return filtered, NMS-deduplicated detections."""
        if conf_threshold is None:
            conf_threshold = CONFIDENCE_THRESHOLD
            
        results = self.yolo_model(img, verbose=False, imgsz=1280)
        detections = []
        if not results or len(results) == 0:
            print("[YOLO] No results returned from model")
            return detections

        h_img, w_img = img.shape[:2]
        total_boxes = sum(len(r.boxes) for r in results)
        print(f"[YOLO] Raw boxes: {total_boxes}, image: {w_img}x{h_img}")
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = self.yolo_model.names[cls_id]
                if cls_id == PERSON_CLASS_ID:
                    continue
                if conf < conf_threshold:
                    print(f"[YOLO] Skipped {cls_name} (conf={conf:.3f} < {conf_threshold})")
                    continue
                print(f"[YOLO] Detected: {cls_name} (conf={conf:.3f})")

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_name = self.yolo_model.names[cls_id]
                detections.append({
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(x2 - x1),
                    "h": int(y2 - y1),
                    "x_pct": float(x1 / w_img * 100),
                    "y_pct": float(y1 / h_img * 100),
                    "w_pct": float((x2 - x1) / w_img * 100),
                    "h_pct": float((y2 - y1) / h_img * 100),
                    "yolo_class": cls_name,
                    "confidence": conf,
                })

        # Apply NMS to remove overlapping detections
        return _nms_detections(detections, NMS_IOU_THRESHOLD)

    def _get_crop(self, img, det, padding=0.15):
        """Crop a detection from the image with padding."""
        h_img, w_img = img.shape[:2]
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w_img, x + w + pad_w)
        y2 = min(h_img, y + h + pad_h)
        return img[y1:y2, x1:x2]

    def _crop_to_bytes(self, crop) -> bytes:
        _, buffer = cv2.imencode('.jpg', crop)
        return buffer.tobytes()

    def _crop_to_b64(self, crop) -> str:
        _, buffer = cv2.imencode('.jpg', crop)
        return base64.b64encode(buffer).decode('utf-8')

    def _generate_embedding(self, image_bytes: bytes) -> list[float] | None:
        """Generate 384-dim DINOv2 embedding locally — no API call."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            inputs = self.processor(images=img, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
            
            # Use the CLS token representation (first token)
            embedding = outputs.last_hidden_state[:, 0, :]
            
            # Normalize for cosine similarity
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().tolist()[0]
        except Exception as e:
            print(f"[ObjectProcessor] DINOv2 embedding error: {e}")
            return None

    def _generate_augmented_embeddings(self, crop) -> list[list[float]]:
        """Generate 5 embeddings from augmented versions of the crop."""
        embeddings = []
        augmented = [
            crop,
            cv2.flip(crop, 1),
            cv2.convertScaleAbs(crop, alpha=1.15, beta=10),
            cv2.convertScaleAbs(crop, alpha=0.85, beta=-10),
            cv2.GaussianBlur(crop, (3, 3), 0),
        ]
        for aug in augmented:
            img_bytes = self._crop_to_bytes(aug)
            emb = self._generate_embedding(img_bytes)
            if emb:
                embeddings.append(emb)
        return embeddings

    def detect_central_object(self, frame_b64: str) -> dict | None:
        """Detect the most central object in the frame (on-demand, COMMAND mode)."""
        img = self._decode_frame(frame_b64)
        if img is None:
            return None

        detections = self._run_yolo(img)
        if not detections:
            return None

        h_img, w_img = img.shape[:2]
        cx_frame, cy_frame = w_img / 2, h_img / 2

        def center_distance(det):
            cx = det["x"] + det["w"] / 2
            cy = det["y"] + det["h"] / 2
            return ((cx - cx_frame) ** 2 + (cy - cy_frame) ** 2) ** 0.5

        # Sort by distance to center, tiebreak by area (larger = better)
        detections.sort(key=lambda d: (center_distance(d), -(d["w"] * d["h"])))
        best = detections[0]

        crop = self._get_crop(img, best)
        best["crop_b64"] = self._crop_to_b64(crop)
        return best

    def recognize_known_objects(self, frame_b64: str) -> list[dict]:
        """Recognize saved objects in the frame (passive, AMBIENT mode)."""
        obj_count = db.objects_count()
        if obj_count == 0:
            print("[ObjectRecog] No objects in ChromaDB — skipping")
            return []

        img = self._decode_frame(frame_b64)
        if img is None:
            return []

        detections = self._run_yolo(img)
        if not detections:
            print("[ObjectRecog] YOLO found 0 detections")
            return []
        print(f"[ObjectRecog] YOLO found {len(detections)} detections, {obj_count} objects in DB")

        matched = []
        for det in detections:
            crop = self._get_crop(img, det)
            img_bytes = self._crop_to_bytes(crop)
            emb = self._generate_embedding(img_bytes)
            if not emb:
                continue

            meta = db.query_object(emb)
            if meta:
                name = meta.get("name", "Unknown")
                obj_info = db.get_object_info(name)
                matched.append({
                    "x": det["x_pct"],
                    "y": det["y_pct"],
                    "w": det["w_pct"],
                    "h": det["h_pct"],
                    "name": name,
                    "notes": obj_info.get("notes", "") if obj_info else "",
                    "dosage": obj_info.get("dosage", "") if obj_info else "",
                    "category": obj_info.get("category", "") if obj_info else "",
                    "confidence": det["confidence"],
                })

        # Deduplicate: if same name appears in overlapping boxes, keep higher confidence
        matched = self._dedup_matched(matched)
        return matched

    def detect_central_yolo_object_preview(self, frame_b64: str) -> list[dict]:
        """Detect the single most central YOLO object (used in COMMAND mode for visibility)."""
        img = self._decode_frame(frame_b64)
        if img is None:
            return []

        detections = self._run_yolo(img)
        if not detections:
            return []

        h_img, w_img = img.shape[:2]
        cx_frame, cy_frame = w_img / 2, h_img / 2

        def center_distance(det):
            cx = det["x"] + det["w"] / 2
            cy = det["y"] + det["h"] / 2
            return ((cx - cx_frame) ** 2 + (cy - cy_frame) ** 2) ** 0.5

        # Sort by distance to center, tiebreak by area (larger = better)
        detections.sort(key=lambda d: (center_distance(d), -(d["w"] * d["h"])))
        best = detections[0]

        return [{
            "x": best["x_pct"],
            "y": best["y_pct"],
            "w": best["w_pct"],
            "h": best["h_pct"],
            "name": best["yolo_class"],
            "notes": "Targeted Object",
            "dosage": "",
            "category": "yolo_raw",
            "confidence": best["confidence"],
        }]

    def detect_all_yolo_debug(self, frame_b64: str) -> list[dict]:
        """Detect all objects using YOLO with very low confidence for debugging."""
        img = self._decode_frame(frame_b64)
        if img is None:
            return []

        # Use 0.2 confidence threshold for debug Mode
        detections = self._run_yolo(img, conf_threshold=0.20)
        if not detections:
            return []

        objects = []
        for det in detections:
            objects.append({
                "x": det["x_pct"],
                "y": det["y_pct"],
                "w": det["w_pct"],
                "h": det["h_pct"],
                "name": f"{det['yolo_class']} ({det['confidence']:.2f})",
                "notes": "Debug Object",
                "dosage": "",
                "category": "yolo_raw",
                "confidence": det["confidence"],
            })
        return objects

    def _dedup_matched(self, matched: list[dict]) -> list[dict]:
        """Remove duplicate matches of the same object name with overlapping boxes."""
        if len(matched) <= 1:
            return matched

        keep = []
        used = set()
        # Sort by confidence descending
        sorted_m = sorted(matched, key=lambda m: m.get("confidence", 0), reverse=True)

        for i, obj in enumerate(sorted_m):
            if i in used:
                continue
            keep.append(obj)
            # Mark overlapping objects with same name as used
            for j in range(i + 1, len(sorted_m)):
                if j in used:
                    continue
                if sorted_m[j]["name"] == obj["name"]:
                    iou = _compute_iou(
                        {"x": obj["x"], "y": obj["y"], "w": obj["w"], "h": obj["h"]},
                        {"x": sorted_m[j]["x"], "y": sorted_m[j]["y"],
                         "w": sorted_m[j]["w"], "h": sorted_m[j]["h"]}
                    )
                    if iou > MATCH_IOU_THRESHOLD:
                        used.add(j)

        return keep

    def register_object(self, frame_b64: str, name: str, notes: str,
                        dosage: str, schedule: str, category: str,
                        object_box: dict) -> dict:
        """Register a new object: crop, embed, store in ChromaDB + JSON."""
        img = self._decode_frame(frame_b64)
        if img is None:
            return {"error": "Invalid image"}

        h_img, w_img = img.shape[:2]
        det = {
            "x": int(object_box["x"] / 100 * w_img),
            "y": int(object_box["y"] / 100 * h_img),
            "w": int(object_box["w"] / 100 * w_img),
            "h": int(object_box["h"] / 100 * h_img),
        }
        crop = self._get_crop(img, det)

        embeddings = self._generate_augmented_embeddings(crop)
        if not embeddings:
            return {"error": "Could not generate embeddings"}

        for emb in embeddings:
            db.add_object_embedding(emb, {"name": name, "category": category})

        # Only add JSON entry if this is a new object (avoid duplicates)
        existing = db.get_object_info(name)
        if not existing:
            db.add_object_info({
                "name": name,
                "notes": notes,
                "dosage": dosage,
                "schedule": schedule,
                "category": category,
            })
        else:
            # Update metadata if re-registering with new info
            updates = {}
            if notes:
                updates["notes"] = notes
            if dosage:
                updates["dosage"] = dosage
            if schedule:
                updates["schedule"] = schedule
            if category:
                updates["category"] = category
            if updates:
                db.update_object_info(name, updates)

        return {"message": f"Object '{name}' registered with {len(embeddings)} embeddings"}

    def get_crop_b64(self, frame_b64: str, box: dict) -> str:
        """Crop a bounding box from a frame and return as base64."""
        img = self._decode_frame(frame_b64)
        if img is None:
            return ""
        h_img, w_img = img.shape[:2]
        det = {
            "x": int(box["x"] / 100 * w_img),
            "y": int(box["y"] / 100 * h_img),
            "w": int(box["w"] / 100 * w_img),
            "h": int(box["h"] / 100 * h_img),
        }
        crop = self._get_crop(img, det)
        return self._crop_to_b64(crop)
