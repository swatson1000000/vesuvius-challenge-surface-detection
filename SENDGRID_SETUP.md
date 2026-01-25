# SendGrid Email Setup

This guide helps you set up SendGrid for training monitor notifications.

## Why SendGrid?

- ✅ Free tier: 100 emails/day (more than enough for :00 and :30 monitoring)
- ✅ No SMTP auth issues like Gmail
- ✅ Simple API authentication
- ✅ Reliable delivery

## Setup Steps

### 1. Create Free SendGrid Account

1. Go to https://sendgrid.com/free
2. Sign up with your email
3. Verify email address
4. Log in to dashboard

### 2. Create API Key

1. Go to https://app.sendgrid.com/settings/api_keys
2. Click **"Create API Key"**
3. Name it: `training-monitor`
4. Select **"Full Access"** (or just "Mail Send")
5. Click **"Create & Use"**
6. **Copy the API key** (you'll only see it once!)

### 3. Store API Key Securely

Add to your `.monitor_env` file:

```bash
export SENDGRID_API_KEY="SG.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Make it secure:
```bash
chmod 600 .monitor_env
```

### 4. Verify Sender Email

SendGrid requires sender email verification for first-time use:

1. Go to https://app.sendgrid.com/settings/sender_auth
2. Click **"Create New Sender"** or use "Single Sender Verification"
3. Use your email: `swatson1000000@gmail.com`
4. Verify by clicking link in email they send you

### 5. Test the Setup

```bash
# Source the environment variable
source .monitor_env

# Run monitor manually
python3 monitor_training.py
```

You should see:
```
✅ Email sent via SendGrid
✅ Status logged to monitor_status.txt
```

### 6. Update Cron Job

The cron job should already source `.monitor_env`, but verify:

```bash
crontab -l
```

Should show:
```
0,30 * * * * source /home/swatson/.../bin/.monitor_env && python3 monitor_training.py >> log/monitor.log 2>&1
```

## Troubleshooting

**"SENDGRID_API_KEY not set"**
- Make sure `.monitor_env` is sourced in cron job
- Check: `echo $SENDGRID_API_KEY` after sourcing

**"SendGrid package not available"**
- Install: `pip install sendgrid`

**"Sender email not verified"**
- Go to https://app.sendgrid.com/settings/sender_auth
- Click verify link in email they send you
- Wait ~5 minutes

**Emails still not arriving**
- Check monitor_status.txt: `tail log/monitor_status.txt`
- Check monitor.log: `tail log/monitor.log`
- Verify API key is correct (copy-paste from SendGrid dashboard again)

## Free Tier Limits

- 100 emails/day
- Monitor at :00 and :30 = 48 emails/day ✅ Well within limit
- No credit card required for 30 days

## Cost After Free Trial

- Business emails: ~$25/month for 20K emails/month
- Your usage: ~1,440 emails/month = well under any paid plan

## API Key Security

⚠️ **Never commit API key to git!**

Already handled:
- `.monitor_env` is in `.gitignore`
- File has `600` permissions (user-only)
