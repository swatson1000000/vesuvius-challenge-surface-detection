#!/usr/bin/env python3
"""
Training Monitor - Emails status updates every 30 minutes using SendGrid
Reads latest training log and sends metrics to email
"""

import os
import re
import glob
import subprocess
import json
import base64
from datetime import datetime
from pathlib import Path

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

# Configuration
LOG_DIR = "/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/log"
RECIPIENT_EMAIL = "swatson1000000@gmail.com"
SENDER_EMAIL = "noreply@trainingmonitor.local"
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")  # Set via environment variable

def get_latest_log():
    """Get the latest training log file"""
    log_files = glob.glob(os.path.join(LOG_DIR, "train_*.log"))
    if not log_files:
        return None
    return max(log_files, key=os.path.getctime)

def parse_training_status(log_file):
    """Parse training log and extract current metrics"""
    if not os.path.exists(log_file):
        return {"error": "Log file not found"}
    
    status = {
        "log_file": os.path.basename(log_file),
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": [],
        "train_losses": [],
        "val_losses": [],
        "latest_epoch": None,
        "best_val_loss": None,
        "best_epoch": None,
        "status": "Unknown"
    }
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract epoch information
        epoch_pattern = r'Epoch (\d+).*?(?:Val - Loss: ([\d.]+))?'
        matches = re.findall(epoch_pattern, content)
        
        if matches:
            # Get latest epoch
            latest = matches[-1]
            status["latest_epoch"] = int(latest[0])
            
            # Extract validation losses
            val_loss_pattern = r'Epoch (\d+).*?Val - Loss: ([\d.]+)'
            val_matches = re.findall(val_loss_pattern, content)
            
            if val_matches:
                status["val_losses"] = [(int(e), float(v)) for e, v in val_matches]
                vals = [v for e, v in val_matches]
                if vals:
                    status["best_val_loss"] = min(vals)
                else:
                    status["best_val_loss"] = None
                # Find epoch with best val loss
                if status["best_val_loss"] is not None:
                    best_idx = min(range(len(status["val_losses"])), 
                                 key=lambda i: status["val_losses"][i][1])
                    status["best_epoch"] = status["val_losses"][best_idx][0]
        
        # Check if training is still running
        if "completed" in content.lower() or "finished" in content.lower():
            status["status"] = "COMPLETED"
        elif status["latest_epoch"] is not None:
            status["status"] = "TRAINING"
        else:
            status["status"] = "INITIALIZING"
        
        # Get last few lines for additional context
        lines = content.strip().split('\n')
        status["last_lines"] = lines[-10:] if len(lines) > 10 else lines
        
    except Exception as e:
        status["error"] = str(e)
    
    return status

def send_email(status):
    """Send training status email via SendGrid"""
    try:
        # Format email content
        if "error" in status:
            subject = f"⚠️ Training Monitor - Error"
            body = f"Error reading training log: {status['error']}"
        else:
            if status['best_val_loss']:
                subject = f"📊 Training Monitor - Epoch {status['latest_epoch']} | Loss: {float(status['best_val_loss']):.4f}"
            else:
                subject = "📊 Training Monitor"
            
            best_loss_str = f"{float(status['best_val_loss']):.6f}" if status['best_val_loss'] else 'N/A'
            best_epoch_str = str(status['best_epoch']) if status['best_epoch'] is not None else 'N/A'
            
            body = f"""
TRAINING STATUS UPDATE
{'='*60}
Time: {status['current_time']}
Log: {status['log_file']}
Status: {status['status']}

METRICS
{'='*60}
Current Epoch: {status['latest_epoch'] if status['latest_epoch'] is not None else 'N/A'}
Best Val Loss: {best_loss_str} @ Epoch {best_epoch_str}

VALIDATION LOSS HISTORY (Last 10)
{'='*60}
"""
            if status['val_losses']:
                # Show last 10 validation losses
                for epoch, val_loss in status['val_losses'][-10:]:
                    body += f"Epoch {epoch:3d}: {val_loss:.6f}\n"
            
            body += f"""
RECENT LOG LINES
{'='*60}
"""
            for line in status['last_lines']:
                body += f"{line}\n"
        
        # Try to send via SendGrid
        email_sent = False
        if SENDGRID_AVAILABLE and SENDGRID_API_KEY:
            try:
                message = Mail(
                    from_email=SENDER_EMAIL,
                    to_emails=RECIPIENT_EMAIL,
                    subject=subject,
                    plain_text_content=body
                )
                sg = SendGridAPIClient(SENDGRID_API_KEY)
                response = sg.send(message)
                if response.status_code == 202:
                    print(f"✅ Email sent via SendGrid")
                    email_sent = True
                else:
                    print(f"⚠️ SendGrid returned status {response.status_code}")
            except Exception as e:
                print(f"⚠️ SendGrid error: {e}")
        else:
            if not SENDGRID_API_KEY:
                print("⚠️ SENDGRID_API_KEY not set - use local logging only")
            else:
                print("⚠️ SendGrid package not available")
        
        # Always log to local status file
        status_file = "/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/log/monitor_status.txt"
        with open(status_file, "w") as f:
            f.write(f"Subject: {subject}\n\n{body}\n")
        
        print(f"✅ Status logged to monitor_status.txt")
        return True
        
    except Exception as e:
        print(f"❌ Error in send_email: {e}")
        return False

def main():
    """Main monitoring function"""
    log_file = get_latest_log()
    
    if not log_file:
        print("❌ No training log found")
        status = {"error": "No training log files found in " + LOG_DIR}
    else:
        print(f"📄 Reading: {log_file}")
        status = parse_training_status(log_file)
        print(f"📊 Status: {status['status']}")
        if status['latest_epoch'] is not None:
            print(f"📈 Current Epoch: {status['latest_epoch']}")
        if status['best_val_loss'] is not None and status['best_epoch'] is not None:
            best_val = float(status['best_val_loss']) if isinstance(status['best_val_loss'], str) else status['best_val_loss']
            print(f"✨ Best Val Loss: {best_val:.6f} @ Epoch {status['best_epoch']}")
        else:
            print(f"⏳ No validation results yet")
    
    # Send email
    if send_email(status):
        print("✅ Monitoring check complete")
    else:
        print("❌ Failed to send email")

if __name__ == "__main__":
    main()
