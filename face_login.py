"""Face login algorithm based on the provided flowchart."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LoginDecision:
    success: bool
    reason: str
    user_id: Optional[str] = None
    confidence: Optional[float] = None


def capture_image() -> bytes:
    """Capture an image from the camera."""
    raise NotImplementedError("Implement camera capture")


def detect_face(image: bytes) -> bool:
    """Return True if a face is detected in the image."""
    raise NotImplementedError("Implement face detection")


def is_real_face(image: bytes) -> bool:
    """Return True if the face passes liveness detection."""
    raise NotImplementedError("Implement liveness detection")


def extract_vector(image: bytes) -> list[float]:
    """Extract a face embedding vector."""
    raise NotImplementedError("Implement embedding extraction")


def load_database_vectors() -> dict[str, list[float]]:
    """Load known user vectors from storage."""
    raise NotImplementedError("Implement database access")


def compare_vectors(
    query_vector: list[float],
    known_vectors: dict[str, list[float]],
) -> tuple[Optional[str], float]:
    """Compare the query to known vectors and return the best match + score."""
    raise NotImplementedError("Implement similarity comparison")


def face_login(confidence_threshold: float = 0.6) -> LoginDecision:
    """Run the face login algorithm."""
    image = capture_image()

    if not detect_face(image):
        return LoginDecision(success=False, reason="No face detected")

    if not is_real_face(image):
        return LoginDecision(success=False, reason="Blocked: fake face detected")

    query_vector = extract_vector(image)
    known_vectors = load_database_vectors()
    user_id, score = compare_vectors(query_vector, known_vectors)

    if score > confidence_threshold and user_id is not None:
        return LoginDecision(
            success=True,
            reason="Login successful",
            user_id=user_id,
            confidence=score,
        )

    return LoginDecision(success=False, reason="Login failed: low confidence", confidence=score)


def main() -> None:
    """Entry point for manual testing."""
    decision = face_login()
    print(decision)


if __name__ == "__main__":
    main()
