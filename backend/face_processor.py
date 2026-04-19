import cv2
import base64
import numpy as np
import random
from deepface import DeepFace
import db

# Minimum face size in pixels — filters out false-positive micro-detections
MIN_FACE_PX = 50
MAX_VECTORS_PER_PERSON = 20
ACCUMULATE_PROBABILITY = 0.10  # 10% chance to add a new vector during scan

class FaceProcessor:
    def __init__(self):
        # Ensure deepface weights are downloaded by doing a dummy init
        try:
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.represent(dummy_img, model_name="Facenet512", enforce_detection=False, detector_backend="mtcnn")
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

    def _generate_augmented_embeddings(self, face_crop):
        """Generate multiple embeddings from augmented versions of the face crop."""
        embeddings = []
        
        # 1. Original
        r = DeepFace.represent(face_crop, model_name="Facenet512", enforce_detection=False, detector_backend="skip")
        if r:
            embeddings.append(r[0]["embedding"])
        
        # 2. Horizontal flip
        flipped = cv2.flip(face_crop, 1)
        r = DeepFace.represent(flipped, model_name="Facenet512", enforce_detection=False, detector_backend="skip")
        if r:
            embeddings.append(r[0]["embedding"])
        
        # 3. Slightly brighter
        bright = cv2.convertScaleAbs(face_crop, alpha=1.15, beta=10)
        r = DeepFace.represent(bright, model_name="Facenet512", enforce_detection=False, detector_backend="skip")
        if r:
            embeddings.append(r[0]["embedding"])
        
        # 4. Slightly darker
        dark = cv2.convertScaleAbs(face_crop, alpha=0.85, beta=-10)
        r = DeepFace.represent(dark, model_name="Facenet512", enforce_detection=False, detector_backend="skip")
        if r:
            embeddings.append(r[0]["embedding"])
        
        # 5. Slight Gaussian blur (simulates motion)
        blurred = cv2.GaussianBlur(face_crop, (3, 3), 0)
        r = DeepFace.represent(blurred, model_name="Facenet512", enforce_detection=False, detector_backend="skip")
        if r:
            embeddings.append(r[0]["embedding"])
        
        return embeddings

    def process_frame(self, frame_b64: str):
        try:
            img_data = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return []

            try:
                results = DeepFace.represent(img, model_name="Facenet512", enforce_detection=False, detector_backend="mtcnn")
            except ValueError:
                return []
            except Exception as e:
                print(f"[FaceProcessor] DeepFace.represent Error: {e}", flush=True)
                return []
                
            detected_faces = []
            if not isinstance(results, list):
                results = [results]
                
            h_img, w_img = img.shape[:2]
            
            # Cap at 4 people at a time
            for face_obj in results[:4]:
                area = face_obj.get("facial_area")
                embedding = face_obj.get("embedding")
                
                if not (area and embedding):
                    continue
                
                fw, fh = area.get('w', 0), area.get('h', 0)
                
                # Skip: no-detection fallback (whole image)
                if fw == w_img and fh == h_img:
                    continue
                
                # Skip: too small — these are false positives
                if fw < MIN_FACE_PX or fh < MIN_FACE_PX:
                    continue
                
                x_pct = float((area['x'] / w_img) * 100)
                y_pct = float((area['y'] / h_img) * 100)
                w_pct = float((fw / w_img) * 100)
                h_pct = float((fh / h_img) * 100)

                # Query ChromaDB for identity
                metadata = db.query_face(embedding)

                if metadata:
                    name = metadata.get("name", "Unknown")
                    relation = metadata.get("relation", "Unset")
                    
                    # Get last conversation topic from patient_context.json
                    topics = db.get_face_topics(name)
                    last = topics.get("last_topic", "") if topics else ""
                    if not last:
                        last = metadata.get("last_conversation", "No record")
                    
                    # Probabilistic vector accumulation for recognized faces
                    if random.random() < ACCUMULATE_PROBABILITY:
                        vec_count = db.count_face_vectors(name)
                        if vec_count < MAX_VECTORS_PER_PERSON:
                            db.add_face(embedding, {
                                "name": name,
                                "relation": relation,
                                "last_conversation": last
                            })
                else:
                    name = "Unknown Person"
                    relation = "N/A"
                    last = "N/A"

                detected_faces.append({
                    "x": x_pct,
                    "y": y_pct,
                    "w": w_pct,
                    "h": h_pct,
                    "name": name,
                    "relation": relation,
                    "lastConversation": last,
                    "topics": (db.get_face_topics(name).get("topics", [])[-3:]) if name != "Unknown Person" else []
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
                face_crop = self._crop_face(img, area)
            else:
                # Fallback: detect faces and pick the largest
                results = DeepFace.represent(img, model_name="Facenet512", enforce_detection=True, detector_backend="mtcnn")
                if not results:
                    return {"error": "No face found"}
                if isinstance(results, list):
                    results.sort(key=lambda x: x["facial_area"]["w"] * x["facial_area"]["h"], reverse=True)
                area = results[0]["facial_area"]
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

