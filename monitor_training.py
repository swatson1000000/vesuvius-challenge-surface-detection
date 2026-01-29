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
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuration
LOG_DIR = "/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/log"
RECIPIENT_EMAIL = "swatson1000000@gmail.com"
SENDER_EMAIL = "swatson1000000@gmail.com"
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")  # Set via .monitor_env

def get_latest_log(custom_log_file=None):
    """Get the latest training log file"""
    # If custom log file specified, use it if it exists
    if custom_log_file and os.path.exists(custom_log_file):
        return custom_log_file
    
    # First, try to read from train_v9_progressive.log (current training)
    primary_log = os.path.join(LOG_DIR, "train_v9_progressive.log")
    if os.path.exists(primary_log):
        return primary_log
    
    # Fallback: search for other training logs sorted by creation time
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
    """Send training status email via Gmail SMTP"""
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
        
        # Send via Gmail SMTP
        email_sent = False
        if GMAIL_APP_PASSWORD:
            try:
                print(f"📧 Sending email via Gmail SMTP...")
                server = smtplib.SMTP("smtp.gmail.com", 587, timeout=15)
                server.starttls()
                server.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
                
                message = MIMEMultipart()
                message["From"] = SENDER_EMAIL
                message["To"] = RECIPIENT_EMAIL
                message["Subject"] = subject
                message.attach(MIMEText(body, "plain"))
                
                server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, message.as_string())
                server.quit()
                
                print(f"✅ Email sent successfully!")
                email_sent = True
            except Exception as e:
                print(f"⚠️ Gmail SMTP failed: {e}")
        else:
            print("⚠️ GMAIL_APP_PASSWORD not set")
        
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
    """Main monitoring function - runs once and exits (designed for cron scheduling)"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Training monitor')
    parser.add_argument('--log-file', type=str, default=None, 
                        help='Path to training log file (default: search in LOG_DIR)')
    args = parser.parse_args()
    
    print(f"📊 Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    log_file = get_latest_log(custom_log_file=args.log_file)
    
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
    
    # Send email and log status
    if send_email(status):
        print("✅ Monitor check complete")
    else:
        print("⚠️ Email send failed")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
