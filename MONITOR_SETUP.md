# Training Monitor Setup Guide

## Overview
Automatically monitors training progress every 30 minutes and emails status updates to your Gmail account.

## Prerequisites

1. **Gmail Account Setup:**
   - Enable 2-Factor Authentication on your Gmail account
   - Generate an "App Password" at https://myaccount.google.com/apppasswords
   - Select "Mail" and "Windows Computer" (or Linux)
   - Copy the 16-character password generated

2. **Python Requirements:** (already installed in phi4 environment)
   - smtplib (standard library)
   - email (standard library)

## Installation

### Option 1: Secure Setup (Recommended)

Uses an environment file to store credentials securely:

```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection
bash setup_monitor_secure.sh
```

This will:
- Create `.monitor_env` file with restricted permissions (600)
- Add cron job to run every 30 minutes
- Log output to `log/monitor.log`

### Option 2: Direct Setup

```bash
cd /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection
bash setup_monitor.sh
```

## Manual Testing

Test the monitor before relying on cron:

```bash
# Set password
export GMAIL_APP_PASSWORD="your_16_char_app_password"

# Run manually
python3 monitor_training.py
```

Expected output:
```
📄 Reading: train_20260125_164944.log
📊 Status: TRAINING
📈 Epoch: 42
✨ Best Val Loss: 0.354321
✅ Email sent to swatson1000000@gmail.com
```

## Email Notifications

You'll receive emails like:

```
Subject: 📊 Training Monitor - Epoch 42 | Loss: 0.3543

TRAINING STATUS UPDATE
============================================================
Time: 2026-01-25 17:30:00
Log: train_20260125_164944.log
Status: TRAINING

METRICS
============================================================
Current Epoch: 42
Best Val Loss: 0.354321 @ Epoch 42

VALIDATION LOSS HISTORY (Last 10)
============================================================
Epoch  35: 0.360000
Epoch  36: 0.358000
Epoch  37: 0.356000
...
```

## Monitoring the Monitor

Check monitor logs:
```bash
tail -f log/monitor.log
```

View all cron jobs:
```bash
crontab -l
```

## Removing the Monitor

Remove cron job:
```bash
crontab -e
# Delete the monitor_training.py line, save and exit
```

Remove environment file (if using secure setup):
```bash
rm /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/.monitor_env
```

## Troubleshooting

### Email not received
1. Check `log/monitor.log` for errors
2. Verify app password is correct
3. Verify GMAIL_APP_PASSWORD is set in cron environment
4. Check Gmail's 2FA settings

### Cron job not running
```bash
# Check if cron service is running
sudo service cron status

# Check cron logs
sudo tail -f /var/log/syslog | grep CRON

# Verify cron job syntax
crontab -l
```

### Parse errors in training log
The monitor will still email with raw last 10 lines if parsing fails.

## Advanced: Customize Email Frequency

To change from 30 minutes to different interval:

```bash
# Edit crontab
crontab -e

# Change the schedule:
# Every 15 minutes: */15 * * * *
# Every hour: 0 * * * *
# Every 6 hours: 0 */6 * * *
# Every day at 9 AM: 0 9 * * *

# For cron syntax help: https://crontab.guru
```

## Files

- `monitor_training.py` - Main monitoring script
- `setup_monitor.sh` - Basic setup (inline password)
- `setup_monitor_secure.sh` - Secure setup (environment file)
- `.monitor_env` - Environment file (created by secure setup, not in git)
- `log/monitor.log` - Monitor execution logs
