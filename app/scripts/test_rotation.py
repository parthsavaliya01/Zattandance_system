import os
import sys
import logging

# Ensure parent path is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.core.config import settings
from app.core.logging_config import DynamicRotatingFileHandler

def test_rotation():
    print("=== Testing DynamicRotatingFileHandler ===")
    test_log_path = "logs/test_rotation_cam_111.log"
    
    # 1. Clean up any previous test logs
    for suffix in ["", "_1", "_2", "_3", "_4", "_5"]:
        name = f"logs/test_rotation_cam_111{suffix}.log"
        if os.path.exists(name):
            os.remove(name)
            
    # 2. Set maxBytes to 50 KB (50 * 1024)
    max_bytes = 50 * 1024
    
    # Create handler
    handler = DynamicRotatingFileHandler(test_log_path, maxBytes=max_bytes)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    
    logger = logging.getLogger("test_rotator")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # 3. Write 150 KB of logs (this should produce 3 log files)
    print("Writing logs to trigger rotation...")
    log_message = "A" * 1000  # ~1 KB per message
    for i in range(150):
        logger.info(f"Message {i:03d}: {log_message}")
        
    handler.close()
    logger.removeHandler(handler)
    
    # 4. Check generated files
    print("\nGenerated files:")
    expected_files = [
        "logs/test_rotation_cam_111.log",
        "logs/test_rotation_cam_111_1.log",
        "logs/test_rotation_cam_111_2.log",
        "logs/test_rotation_cam_111_3.log",
    ]
    
    for f in expected_files:
        if os.path.exists(f):
            print(f"  [FOUND] {f} - Size: {os.path.getsize(f) / 1024:.2f} KB")
        else:
            print(f"  [MISSING] {f}")
            
    # 5. Test Startup behavior: do not overwrite existing files, append or find next index
    print("\nTesting startup behavior with existing full files...")
    # Re-open handler
    handler2 = DynamicRotatingFileHandler(test_log_path, maxBytes=max_bytes)
    handler2.setFormatter(formatter)
    logger.addHandler(handler2)
    
    # Write a new message
    logger.info("New message on startup!")
    handler2.close()
    
    # Check if a new file logs/test_rotation_cam_111_3.log or logs/test_rotation_cam_111_4.log was created
    for f in expected_files:
        if os.path.exists(f):
            print(f"  {f} - Size: {os.path.getsize(f) / 1024:.2f} KB")
            
    # 6. Cleanup
    print("\nCleaning up test logs...")
    for suffix in ["", "_1", "_2", "_3", "_4", "_5"]:
        name = f"logs/test_rotation_cam_111{suffix}.log"
        if os.path.exists(name):
            os.remove(name)
            
    print("Test finished successfully!")

if __name__ == "__main__":
    test_rotation()
