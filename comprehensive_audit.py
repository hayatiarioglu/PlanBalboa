import py_compile
import traceback
import sys
from pathlib import Path

def audit_all_files():
    root = Path("aether")
    py_files = list(root.rglob("*.py"))
    
    print(f"Found {len(py_files)} Python files in {root.resolve()} to audit.\n")
    
    compilation_errors = []
    
    for file_path in py_files:
        try:
            py_compile.compile(str(file_path), doraise=True)
            print(f"[OK] {file_path}")
        except py_compile.PyCompileError as e:
            print(f"[SYNTAX ERROR] {file_path}: {e}")
            compilation_errors.append((file_path, str(e)))
        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")
            compilation_errors.append((file_path, str(e)))
            
    print("\n--- SYNTAX / COMPILATION SUMMARY ---")
    if compilation_errors:
        print(f"FAIL: Found {len(compilation_errors)} compilation errors!")
        for path, err in compilation_errors:
            print(f"  - {path}: {err}")
    else:
        print("SUCCESS: 100% of files passed python syntax & compilation checks!")

if __name__ == "__main__":
    audit_all_files()
