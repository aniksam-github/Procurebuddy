import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing command"}))
        return 1

    command = sys.argv[1]
    repo_root = Path(__file__).resolve().parent.parent
    backend_dir = repo_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    try:
        if command == "ask":
            payload = json.loads(sys.stdin.read() or "{}")
            from core import ask_question

            message = payload.get("message", "")
            history = payload.get("history") or []
            with redirect_stdout(StringIO()):
                reply = ask_question(message, history)
            print(json.dumps({"reply": reply}, ensure_ascii=False))
            return 0

        if command == "reindex":
            from ingest import create_vector_db

            with redirect_stdout(StringIO()):
                result = create_vector_db()
            print(json.dumps(result, ensure_ascii=False))
            return 0

        print(json.dumps({"error": f"Unsupported command: {command}"}))
        return 1
    except Exception as exc:  # pragma: no cover - bridge error path
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
