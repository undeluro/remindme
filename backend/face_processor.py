import cv2
import base64
import numpy as np
import random
import threading
from collections import Counter, deque
from deepface import DeepFace
import db

# YuNet is ~6x faster than MTCNN on CPU (≈90ms vs ≈515ms per 960x540 frame)
DETECTOR = "yunet"
MODEL = "Facenet512"

# Minimum face size in pixels — filters out false-positive micro-detections
MIN_FACE_PX = 50
MAX_FACES = 4
MAX_VECTORS_PER_PERSON = 20
ACCUMULATE_PROBABILITY = 0.10  # 10% chance to add a new vector during scan
ACCUMULATE_MAX_DISTANCE = 0.22  # only learn from very confident matches, so wrong matches can't pollute a person

UNKNOWN = "Unknown Person"

# DeepFace's detectors (YuNet in particular) are not thread-safe: the face loop and a
# registration request calling it at the same time crashes the whole process.
_deepface_lock = threading.Lock()


def _represent(img, **kwargs):
    with _deepface_lock:
        return DeepFace.represent(img, model_name=MODEL, enforce_detection=False, **kwargs)


class FaceProcessor:
    def __init__(self):
        # Ensure deepface weights are downloaded by doing a dummy init
        try:
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            _represent(dummy_img, detector_backend=DETECTOR)
        except Exception:
            pass

    def _crop_face(self, img, area, padding=0.3):
        """Crop a face region from the image with some padding for context."""
        h_img, w_img = img.shape[:2]
        x, y, w, h = area['x'], area['y'], area['w'], area['h']

        pad_w = int(w * padding)
        pad_h = int(h * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w_img, x + w + pad_w)
        y2 = min(h_img, y + h + pad_h)

        return img[y1:y2, x1:x2]

    def _detect_and_embed(self, img):
        """Detect + align faces and embed them. Returns real detections only, largest first."""
        try:
            results = _represent(img, detector_backend=DETECTOR)
        except ValueError:
            return []
        if not isinstance(results, list):
            results = [results]

        h_img, w_img = img.shape[:2]
        faces = []
        for face_obj in results:
            area = face_obj.get("facial_area")
            embedding = face_obj.get("embedding")
            if not (area and embedding):
                continue
            fw, fh = area.get('w', 0), area.get('h', 0)
            # Skip: no-detection fallback (whole image)
            if fw == w_img and fh == h_img:
                continue
            faces.append((area, embedding))

        faces.sort(key=lambda f: f[0]['w'] * f[0]['h'], reverse=True)
        return faces

    def _embed_crop(self, crop):
        """Embed the main face in a crop using the same detect+align path as recognition."""
        faces = self._detect_and_embed(crop)
        if faces:
            return faces[0][1]
        # Detector missed the face in the crop — fall back to embedding the crop as-is
        r = _represent(crop, detector_backend="skip")
        return r[0]["embedding"] if r else None

    def _generate_augmented_embeddings(self, face_crop):
        """Generate multiple embeddings from augmented versions of the face crop."""
        augmented = [
            face_crop,                                              # 1. Original
            cv2.flip(face_crop, 1),                                 # 2. Horizontal flip
            cv2.convertScaleAbs(face_crop, alpha=1.15, beta=10),    # 3. Slightly brighter
            cv2.convertScaleAbs(face_crop, alpha=0.85, beta=-10),   # 4. Slightly darker
            cv2.GaussianBlur(face_crop, (3, 3), 0),                 # 5. Slight blur (simulates motion)
        ]
        embeddings = []
        for aug in augmented:
            emb = self._embed_crop(aug)
            if emb:
                embeddings.append(emb)
        return embeddings

    def process_frame(self, frame_b64: str):
        try:
            img_data = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return []

            try:
                faces = self._detect_and_embed(img)
            except Exception as e:
                print(f"[FaceProcessor] DeepFace.represent Error: {e}", flush=True)
                return []

            detected_faces = []
            h_img, w_img = img.shape[:2]

            # Largest faces first, so people in the background crowd don't push out the person in front
            for area, embedding in faces:
                if len(detected_faces) >= MAX_FACES:
                    break

                fw, fh = area.get('w', 0), area.get('h', 0)

                # Skip: too small — these are false positives
                if fw < MIN_FACE_PX or fh < MIN_FACE_PX:
                    continue

                x_pct = float((area['x'] / w_img) * 100)
                y_pct = float((area['y'] / h_img) * 100)
                w_pct = float((fw / w_img) * 100)
                h_pct = float((fh / h_img) * 100)

                # Query ChromaDB for identity (voting over nearest vectors)
                metadata, distance = db.match_face(embedding)

                topics_list = []
                if metadata:
                    name = metadata.get("name", "Unknown")
                    relation = metadata.get("relation", "Unset")

                    # Get last conversation topic from patient_context.json
                    topics = db.get_face_topics(name) or {}
                    topics_list = topics.get("topics", [])[-3:]
                    last = topics.get("last_topic", "")  # empty until there has been a conversation — the tag then shows no topic line

                    # Probabilistic vector accumulation — only for very confident matches
                    if distance < ACCUMULATE_MAX_DISTANCE and random.random() < ACCUMULATE_PROBABILITY:
                        vec_count = db.count_face_vectors(name)
                        if vec_count < MAX_VECTORS_PER_PERSON:
                            db.add_face(embedding, {
                                "name": name,
                                "relation": relation,
                                "last_conversation": last
                            })
                else:
                    name = UNKNOWN
                    relation = ""
                    last = ""

                detected_faces.append({
                    "x": x_pct,
                    "y": y_pct,
                    "w": w_pct,
                    "h": h_pct,
                    "name": name,
                    "relation": relation,
                    "lastConversation": last,
                    "topics": topics_list
                })

            return detected_faces
        except Exception as e:
            print(f"FaceProcessor general exception: {str(e)}", flush=True)
            return []

    def register_new_face(self, frame_b64: str, name: str, relation: str, face_box: dict = None):
        try:
            img_data = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {"error": "Invalid image format"}

            h_img, w_img = img.shape[:2]

            if face_box:
                # Convert percentage-based box to pixel coordinates
                px_x = int(face_box["x"] / 100 * w_img)
                px_y = int(face_box["y"] / 100 * h_img)
                px_w = int(face_box["w"] / 100 * w_img)
                px_h = int(face_box["h"] / 100 * h_img)
                area = {"x": px_x, "y": px_y, "w": px_w, "h": px_h}
            else:
                # Fallback: detect faces and pick the largest
                faces = self._detect_and_embed(img)
                if not faces:
                    return {"error": "No face found"}
                area = faces[0][0]

            # Padded crop; each embedding re-detects and aligns the face inside it,
            # so stored vectors match what process_frame produces at recognition time
            face_crop = self._crop_face(img, area)

            # Generate multiple augmented embeddings for robust registration
            embeddings = self._generate_augmented_embeddings(face_crop)
            if not embeddings:
                return {"error": "Could not compute face embeddings from crop"}

            metadata = {
                "name": name,
                "relation": relation,
                "last_conversation": "No conversational history yet."
            }

            # Store all augmented embeddings
            for emb in embeddings:
                db.add_face(emb, metadata)

            # Encode cropped face as base64 for frontend preview
            _, buffer = cv2.imencode('.jpg', face_crop)
            face_b64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "message": f"Successfully registered {name} ({len(embeddings)} vectors stored)",
                "face_preview": f"data:image/jpeg;base64,{face_b64}"
            }

        except ValueError:
            return {"error": "Could not detect a face in the frame"}
        except Exception as e:
            return {"error": str(e)}


def box_iou(a, b):
    """IoU of two {x, y, w, h} boxes (any units, as long as both use the same)."""
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2 = min(a["x"] + a["w"], b["x"] + b["w"])
    iy2 = min(a["y"] + a["h"], b["y"] + b["h"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """Smooths identities across frames so name tags don't flicker or jump between people.

    Each face is matched to a track from the previous frames by box overlap; the
    displayed name is the majority vote over the track's recent recognitions.
    """

    HISTORY = 5         # recognitions remembered per track
    MATCH_IOU = 0.25    # min overlap to treat a face as the same person as a track
    MAX_MISSED = 1      # keep showing a track for this many frames after it disappears

    def __init__(self):
        self.tracks = []
        self._next_id = 0

    def _voted_name(self, track):
        return Counter(track["names"]).most_common(1)[0][0]

    def update(self, faces: list[dict]) -> list[dict]:
        unmatched = list(range(len(self.tracks)))
        for face in faces:
            best, best_iou = None, self.MATCH_IOU
            for ti in unmatched:
                iou = box_iou(face, self.tracks[ti]["face"])
                if iou >= best_iou:
                    best, best_iou = ti, iou
            if best is None and face["name"] != UNKNOWN:
                # Person moved too far for the boxes to overlap — continue their existing track
                for ti in unmatched:
                    if self._voted_name(self.tracks[ti]) == face["name"]:
                        best = ti
                        break
            if best is None:
                self.tracks.append({"id": self._next_id, "face": face, "names": deque(maxlen=self.HISTORY), "by_name": {}, "missed": 0})
                self._next_id += 1
                track = self.tracks[-1]
            else:
                unmatched.remove(best)
                track = self.tracks[best]
                track["face"] = face
                track["missed"] = 0
            track["names"].append(face["name"])
            track["by_name"][face["name"]] = face

        for ti in unmatched:
            self.tracks[ti]["missed"] += 1
        self.tracks = [t for t in self.tracks if t["missed"] <= self.MAX_MISSED]

        # A known person can only be in one place: keep their freshest track
        output = []
        shown_names = set()
        for track in sorted(self.tracks, key=lambda t: t["missed"]):
            name = self._voted_name(track)
            if name != UNKNOWN and name in shown_names:
                continue
            shown_names.add(name)
            # Identity info from the voted name, position from the latest box
            shown = dict(track["by_name"][name])
            for key in ("x", "y", "w", "h"):
                shown[key] = track["face"][key]
            shown["track_id"] = track["id"]
            output.append(shown)
        return output
