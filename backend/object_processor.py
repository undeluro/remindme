import cv2
import base64
import io
import re
import threading
from difflib import SequenceMatcher
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from ultralytics import YOLO
import db

# YOLOE detects whatever words it is given (open vocabulary), unlike plain YOLO's fixed 80 COCO classes
DETECTOR_WEIGHTS = "yoloe-26s-seg.pt"
DETECTOR_IMGSZ = 640
# Household things worth remembering. The detector type of every saved object is added on top.
# "person" is included only so people are labelled as people and then skipped, not mistaken for objects.
BASE_VOCABULARY = [
    "person", "cell phone", "cup", "mug", "bottle", "drink can", "glasses", "sunglasses", "keys", "wallet",
    "pillow", "cushion", "blanket", "remote control", "laptop", "book", "notebook", "pen",
    "bag", "backpack", "handbag", "hat", "shoe", "slipper", "watch", "medicine box", "pill bottle",
    "blister pack", "toothbrush", "comb", "plate", "bowl", "spoon", "fork", "knife", "scissors",
    "headphones", "charger", "toy", "teddy bear", "plush toy", "umbrella", "clock", "lamp",
    "plant", "vase", "photo frame", "box",
]

CONFIDENCE_THRESHOLD = 0.4  # Lowered to 0.4 to pick up smaller objects
COMMAND_CONFIDENCE_THRESHOLD = 0.25  # Command mode: the user is deliberately showing something, so accept weaker detections
MAX_AMBIENT_CANDIDATES = 6  # Ambient mode: only compare the largest detections with saved objects
NMS_IOU_THRESHOLD = 0.5  # IoU threshold for deduplicating YOLO detections
MATCH_IOU_THRESHOLD = 0.3  # IoU threshold for deduplicating matched objects

# Command mode only looks at things in the middle of the frame (fractions of width/height).
# Keep in sync with .center-zone in frontend/src/index.css
CENTER_ZONE = (0.25, 0.15, 0.75, 0.85)  # x1, y1, x2, y2


def in_center_zone(x_pct: float, y_pct: float, w_pct: float, h_pct: float) -> bool:
    """True if the center of a box (given in % of the frame) lies inside CENTER_ZONE."""
    cx = (x_pct + w_pct / 2) / 100
    cy = (y_pct + h_pct / 2) / 100
    x1, y1, x2, y2 = CENTER_ZONE
    return x1 <= cx <= x2 and y1 <= cy <= y2


LABEL_MATCH_MIN = 0.6  # how well the read text must match a saved label to count as that object
SHOT_HINT_MARGIN = 0.25  # saving: the object may move this much (fraction of its size) between photos
SHOT_MIN_IOU = 0.3  # saving: a detection must overlap the object's last known box this much to be used


def object_payload(name: str) -> dict:
    """What the frontend tag shows for a saved object."""
    info = db.get_object_info(name) or {}
    label = " ".join(p for p in (info.get("label_name", ""), info.get("label_dosage", "")) if p)
    return {
        "name": name,
        "notes": info.get("notes", ""),
        "dosage": info.get("dosage", ""),
        "category": info.get("category", ""),
        "label": label,
    }


def _label_tokens(text: str) -> list[str]:
    """Meaningful words/numbers of a label: lowercase, letters+digits only, no 1-2 letter noise."""
    tokens = re.findall(r"[^\W_]+", (text or "").lower())
    return [t for t in tokens if len(t) >= 3 or t.isdigit()]


def label_match_score(saved: dict, reading: dict) -> float:
    """0..1 — how well a freshly read label matches the label saved with an object.

    Scored on the saved product name + dosage (falling back to its full text), each word
    fuzzily matched against the words just read, so small OCR slips don't matter.
    """
    key = " ".join(p for p in (saved.get("label_name", ""), saved.get("label_dosage", "")) if p) or saved.get("label_text", "")
    key_tokens = _label_tokens(key)
    read_tokens = _label_tokens(" ".join(reading.get(k, "") for k in ("text", "product_name", "dosage")))
    if not key_tokens or not read_tokens:
        return 0.0
    total = 0.0
    for kt in key_tokens:
        best = max(SequenceMatcher(None, kt, rt).ratio() for rt in read_tokens)
        # numbers (dosages) must match exactly — 200 mg is not 400 mg
        total += (1.0 if kt in read_tokens else 0.0) if kt.isdigit() else best
    return total / len(key_tokens)


def resolve_by_label(candidates: list[str], reading: dict) -> tuple[str | None, bool]:
    """Decide between look-alike objects using the label text just read.

    Returns (name, keep): the saved object whose label matches best, or the look-alike
    winner when the text doesn't help; keep=False means the text proves it's none of
    the saved labelled objects (e.g. a different medicine), so no tag should be shown.
    """
    if not candidates:
        return None, False
    if not any(reading.get(k) for k in ("text", "product_name", "dosage")):
        return candidates[0], True  # nothing readable — trust appearance

    infos = {name: db.get_object_info(name) or {} for name in candidates}
    labelled = {n: i for n, i in infos.items() if i.get("label_text") or i.get("label_name")}
    if not labelled:
        return candidates[0], True

    scores = {n: label_match_score(i, reading) for n, i in labelled.items()}
    best_name = max(scores, key=scores.get)
    if scores[best_name] >= LABEL_MATCH_MIN:
        return best_name, True
    if candidates[0] in labelled:
        return None, False  # looks like a saved box, but the text says it's something else
    return candidates[0], True


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
        # Inference and vocabulary updates come from different threads (object loop, save requests)
        self._yolo_lock = threading.Lock()
        self._dino_lock = threading.Lock()
        self.yolo_model = YOLO(DETECTOR_WEIGHTS)
        self.refresh_vocabulary()

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

    def refresh_vocabulary(self):
        """Tell the detector which words to look for: the base list plus the type of every saved object."""
        words = list(BASE_VOCABULARY)
        for obj in db.list_objects():
            word = (obj.get("detector_class") or obj.get("name") or "").strip().lower()
            if word and word not in words:
                words.append(word)
        with self._yolo_lock:
            self.yolo_model.set_classes(words, self.yolo_model.get_text_pe(words))
        print(f"[ObjectProcessor] Detector vocabulary: {len(words)} words")

    def _run_yolo(self, img, conf_threshold=None):
        """Run YOLO and return filtered, NMS-deduplicated detections."""
        if conf_threshold is None:
            conf_threshold = CONFIDENCE_THRESHOLD

        with self._yolo_lock:
            results = self.yolo_model(img, verbose=False, imgsz=DETECTOR_IMGSZ)
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
                if cls_name == "person":
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
            with self._dino_lock, torch.no_grad():
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

    def _central_detections(self, img) -> list[dict]:
        """YOLO detections whose center lies inside CENTER_ZONE (command mode)."""
        return [
            d for d in self._run_yolo(img, conf_threshold=COMMAND_CONFIDENCE_THRESHOLD)
            if in_center_zone(d["x_pct"], d["y_pct"], d["w_pct"], d["h_pct"])
        ]

    def detect_central_object(self, frame_b64: str) -> dict | None:
        """Detect the most central object in the frame (on-demand, COMMAND mode)."""
        img = self._decode_frame(frame_b64)
        if img is None:
            return None

        detections = self._central_detections(img)
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

        # The bigger vocabulary finds more things — only embed the largest few to keep each scan fast
        detections.sort(key=lambda d: d["w"] * d["h"], reverse=True)
        matched = []
        for det in detections[:MAX_AMBIENT_CANDIDATES]:
            crop = self._get_crop(img, det)
            img_bytes = self._crop_to_bytes(crop)
            emb = self._generate_embedding(img_bytes)
            if not emb:
                continue

            candidates = db.query_object_candidates(emb)
            if candidates:
                obj = object_payload(candidates[0]["name"])
                obj.update({
                    "x": det["x_pct"],
                    "y": det["y_pct"],
                    "w": det["w_pct"],
                    "h": det["h_pct"],
                    "confidence": det["confidence"],
                    # Internal (stripped before sending): every look-alike, and the photo for reading its label
                    "_candidates": [c["name"] for c in candidates],
                    "_crop_jpeg": img_bytes,
                })
                matched.append(obj)

        # Deduplicate: if same name appears in overlapping boxes, keep higher confidence
        matched = self._dedup_matched(matched)
        return matched

    def detect_central_yolo_object_preview(self, frame_b64: str) -> list[dict]:
        """Detect the single most central YOLO object (used in COMMAND mode for visibility)."""
        img = self._decode_frame(frame_b64)
        if img is None:
            return []

        detections = self._central_detections(img)
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
            "targeted": True,  # the frontend shows "+ Save" on this box
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

    def prepare_shots(self, frames_b64: list[str], hint_box: dict | None) -> list[dict]:
        """Find and crop the object in each of several frames (taken while the user turns it).

        The object moves between frames, so it is re-detected in every frame. With a hint_box, only
        detections centred inside it (enlarged a bit for movement) and overlapping it well count —
        otherwise hint_box itself is used, never some other object. Without a hint, the most central one.
        Returns [{"crop_b64", "box"}] with box in % of the frame.
        """
        shots = []
        for frame_b64 in frames_b64:
            img = self._decode_frame(frame_b64)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            detections = self._central_detections(img)

            best = None
            if hint_box:
                hint_px = {"x": hint_box["x"] / 100 * w_img, "y": hint_box["y"] / 100 * h_img,
                           "w": hint_box["w"] / 100 * w_img, "h": hint_box["h"] / 100 * h_img}
                mx, my = hint_px["w"] * SHOT_HINT_MARGIN, hint_px["h"] * SHOT_HINT_MARGIN

                def centred_in_hint(d):
                    cx, cy = d["x"] + d["w"] / 2, d["y"] + d["h"] / 2
                    return (hint_px["x"] - mx <= cx <= hint_px["x"] + hint_px["w"] + mx
                            and hint_px["y"] - my <= cy <= hint_px["y"] + hint_px["h"] + my)

                near = [d for d in detections if centred_in_hint(d) and _compute_iou(d, hint_px) >= SHOT_MIN_IOU]
                if near:
                    best = max(near, key=lambda d: _compute_iou(d, hint_px))
            elif detections:
                cx, cy = w_img / 2, h_img / 2
                best = min(detections, key=lambda d: (d["x"] + d["w"] / 2 - cx) ** 2 + (d["y"] + d["h"] / 2 - cy) ** 2)

            if best is not None:
                box = {"x": best["x_pct"], "y": best["y_pct"], "w": best["w_pct"], "h": best["h_pct"]}
            elif hint_box:
                box = dict(hint_box)
            else:
                continue

            det = {"x": int(box["x"] / 100 * w_img), "y": int(box["y"] / 100 * h_img),
                   "w": int(box["w"] / 100 * w_img), "h": int(box["h"] / 100 * h_img)}
            crop = self._get_crop(img, det)
            if crop.size == 0:
                continue
            shots.append({"crop_b64": self._crop_to_b64(crop), "box": box})
        return shots

    def register_object_shots(self, crops_b64: list[str], name: str, info: dict) -> dict:
        """Register an object from several photos: every photo is stored with its 4 variants
        (mirrored, brighter, darker, blurred), so it's recognized from more angles and in more light."""
        embeddings = []
        for crop_b64 in crops_b64:
            crop = self._decode_frame(crop_b64)
            if crop is None or crop.size == 0:
                continue
            embeddings.extend(self._generate_augmented_embeddings(crop))
        if not embeddings:
            return {"error": "Could not generate embeddings"}
        return self._store_object(name, embeddings, info, photos=len(crops_b64))

    def register_object(self, frame_b64: str, name: str, notes: str,
                        dosage: str, schedule: str, category: str,
                        object_box: dict, detector_class: str = "") -> dict:
        """Register a new object from a single frame + box: crop, embed, store in ChromaDB + JSON."""
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

        info = {"notes": notes, "dosage": dosage, "schedule": schedule,
                "category": category, "detector_class": detector_class}
        return self._store_object(name, embeddings, info, photos=1)

    def _store_object(self, name: str, embeddings: list, info: dict, photos: int) -> dict:
        """Save embeddings to ChromaDB and metadata (notes, label text, ...) to patient_context.json."""
        for emb in embeddings:
            db.add_object_embedding(emb, {"name": name, "category": info.get("category", "")})

        # Only add JSON entry if this is a new object (avoid duplicates)
        existing = db.get_object_info(name)
        if not existing:
            db.add_object_info({"name": name, **{k: v or "" for k, v in info.items()}})
        else:
            # Re-registering adds photos; update metadata only with fields that were filled in
            updates = {k: v for k, v in info.items() if v}
            if updates:
                db.update_object_info(name, updates)

        # Make sure the detector looks for this kind of object from now on
        self.refresh_vocabulary()

        return {"message": f"Object '{name}' saved from {photos} photo(s) ({len(embeddings)} embeddings)"}

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
