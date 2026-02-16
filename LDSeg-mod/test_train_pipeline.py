"""
Verification script for training pipeline.
1. Creates dummy data.
2. Updates train_config.ini to point to dummy data and limit epochs to 1.
3. Runs train.py.
4. Checks for output logs/checkpoints.
5. Clean up.
"""
import os
import shutil
import subprocess
import configparser
import numpy as np
from skimage import io

TEMP_DATA_DIR = 'temp_lidc_train_data'
TEMP_LOG_DIR = 'temp_training_logs'
CONFIG_FILE = 'train_config_test.ini'

def create_dummy_data():
    sample_dir = os.path.join(TEMP_DATA_DIR, 'sample_001')
    os.makedirs(sample_dir, exist_ok=True)
    img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    files = {
        'image_001.png': img,
        'label0_001.png': (img > 100).astype(np.uint8) * 255,
        'label1_001.png': (img > 100).astype(np.uint8) * 255, # Duplicate just to satisfy 5
        'label2_001.png': (img > 100).astype(np.uint8) * 255,
        'label3_001.png': (img > 100).astype(np.uint8) * 255,
    }
    for fname, data in files.items():
        io.imsave(os.path.join(sample_dir, fname), data, check_contrast=False)
    print(f"Created dummy data in {TEMP_DATA_DIR}")

def create_test_config():
    # Read template config
    config = configparser.ConfigParser()
    config.read('train_config.ini')
    
    # Modify for testing
    config['Data']['DatasetDir'] = TEMP_DATA_DIR
    config['Data']['ValidationDir'] = TEMP_DATA_DIR
    config['Logging']['LogDir'] = TEMP_LOG_DIR
    config['Training']['Epochs'] = '1'
    config['Training']['BatchSize'] = '1'
    config['Logging']['SaveInterval'] = '1'
    config['Data']['NumWorkers'] = '0' # Avoid multiprocessing issues in test
    
    with open(CONFIG_FILE, 'w') as f:
        config.write(f)
    print(f"Created test config {CONFIG_FILE}")

def run_training():
    print("Running training script...")
    cmd = ['python', 'train.py', '--config', CONFIG_FILE]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        print("Training script finished successfully.")
        print(output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(f"Training failed with code {e.returncode}")
        print(e.output.decode('utf-8'))
        raise e

def verify_output():
    if not os.path.exists(TEMP_LOG_DIR):
        raise FileNotFoundError(f"Log dir {TEMP_LOG_DIR} not found")
    
    logs = os.listdir(TEMP_LOG_DIR)
    print(f"Testing logs found: {logs}")
    
    has_checkpoint = any(f.startswith('checkpoint') for f in logs)
    has_log = 'train.log' in logs
    
    if has_checkpoint and has_log:
        print("Verification SUCCESS: Checkpoints and logs created.")
    else:
        print("Verification FAILED: Missing checkpoints or logs.")

def cleanup():
    if os.path.exists(TEMP_DATA_DIR): shutil.rmtree(TEMP_DATA_DIR)
    if os.path.exists(TEMP_LOG_DIR): shutil.rmtree(TEMP_LOG_DIR)
    if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
    print("Cleanup done.")

if __name__ == "__main__":
    try:
        cleanup() # Pre-clean
        create_dummy_data()
        create_test_config()
        run_training()
        verify_output()
    except Exception as e:
        print(f"Test FAILED with error: {e}")
    finally:
        cleanup()
