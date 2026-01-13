# Terminal Commands - Quick Reference

## 🎯 Commands to Run Now

### TERMINAL 2 - Start API (Copy & Paste This)
```bash
cd "/Users/aryan/Desktop/JHATU RUSHIKESH/Codebase"
uvicorn api:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Will watch for changes in these directories: ['/Users/aryan/Desktop/JHATU RUSHIKESH/Codebase']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
✓ Model loaded successfully
✓ Using 14 features
INFO:     Application startup complete.
```

**Keep this terminal running!**

---

### TERMINAL 3 - Start Streamlit (Copy & Paste This)
```bash
cd "/Users/aryan/Desktop/JHATU RUSHIKESH/Codebase"
streamlit run app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Browser will open automatically!**

**Keep this terminal running!**

---

### TERMINAL 4 - Test Everything (Optional)
```bash
cd "/Users/aryan/Desktop/JHATU RUSHIKESH/Codebase"
python3 test_api.py
```

**This runs once and exits. You should see:**
```
🧪 STARTING API TESTS

Testing /health endpoint...
Status Code: 200
Response: {
  "status": "healthy",
  "model_loaded": true,
  "features_count": 14,
  "api_version": "1.0.0"
}

✅ All tests passed! API is working correctly.
```

---

## 🌐 URLs to Visit

| Service | URL | Description |
|---------|-----|-------------|
| **Streamlit UI** | http://localhost:8501 | Main web interface |
| **API Health** | http://localhost:8000/health | Check API status |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **API Root** | http://localhost:8000 | API information |
| **API Metrics** | http://localhost:8000/metrics | Model performance |

---

## ⌨️ Keyboard Shortcuts

**To Stop Servers:**
- Press `Ctrl + C` in the terminal

**To Restart:**
- Press `Ctrl + C` to stop
- Press `Up Arrow` to get previous command
- Press `Enter` to run again

---

## 📸 Screenshots to Take (For PPT)

1. **Streamlit UI** - Main page with form
2. **Prediction Results** - After clicking "Predict Churn Risk"
3. **API Docs** - Visit http://localhost:8000/docs
4. **Model Metrics** - In Streamlit, go to Performance tab

---

## ✅ Checklist

- [ ] Terminal 2 running (API server)
- [ ] Terminal 3 running (Streamlit)
- [ ] Browser opened to http://localhost:8501
- [ ] "API Status: Online" shows in Streamlit sidebar
- [ ] Can make predictions successfully
- [ ] Screenshots taken for PPT

---

## 🐛 Common Issues

**"Address already in use"**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn api:app --reload --port 8001
```

**"Model files not found"**
```bash
# Check files exist
ls -lh *.pkl

# Re-run training if needed
python3 train_model.py
```

**"API Status: Offline" in Streamlit**
- Make sure Terminal 2 (API) is still running
- Visit http://localhost:8000/health to check
- Restart API server if needed

---

## 📝 Next Steps After Testing

1. ✅ Verify everything works
2. ✅ Take screenshots
3. ⏳ Create PPT presentation
4. ⏳ Add screenshots to PPT
5. ⏳ Create submission package

---

**Keep this file open for quick reference!**
