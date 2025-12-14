# 🌐 Flask Web Demo - Quick Guide

## 🚀 Launch the Web Demo

### 1. Install Flask (if not already installed)
```bash
pip install flask
```

### 2. Start the Web Server
```bash
python app.py
```

### 3. Open in Browser
Navigate to: **http://localhost:5000**

---

## 🎮 Features

### Interactive Controls
- **Scenario Selection**: Choose from Random, Head-On, or Crossing trajectories
- **AI Toggle**: Compare AI-enabled vs coast-only behavior
- **Real-time Visualization**: See satellite trajectories as they avoid debris

### Available Actions

1. **▶️ Run Scenario**
   - Executes a single debris avoidance scenario
   - Shows live trajectory plot
   - Displays metrics: distance, fuel, success rate

2. **📊 Compare AI vs No-AI**
   - Runs same scenario twice (with/without AI)
   - Side-by-side visualization
   - Proves AI effectiveness

3. **⚡ Benchmark Inference**
   - Measures decision latency (100 samples)
   - Compares to ground control (600ms)
   - Shows speedup factor

---

## 📊 What You'll See

### Dashboard Metrics
- **600x Faster**: Edge-AI vs ground control
- **<1ms Latency**: Real-time decision making
- **44KB Model**: Tiny footprint for satellites
- **100% Success**: Demo performance

### Visualizations
- **2D Trajectory Plot**: LVLH orbital frame
- **Distance vs Time**: Safety margin tracking
- **Comparison View**: AI effectiveness
- **Benchmark Dashboard**: Latency statistics

---

## 🎤 Demo Flow for Presentation

### Opening (30 seconds)
1. Open browser to `http://localhost:5000`
2. Explain the Kessler Syndrome threat (shown on page)
3. Highlight key metrics in dashboard

### Live Demo (2 minutes)
```
1. Select "Head-On Collision Course"
2. Check "Use AI Avoidance" ✓
3. Click "▶️ Run Scenario"
4. Watch real-time avoidance!
5. Point out: ✅ No collision, minimal fuel
```

### Comparison (1 minute)
```
1. Click "📊 Compare AI vs No-AI"
2. Show side-by-side trajectories
3. Highlight: AI avoids, No-AI collides
```

### Edge Performance (1 minute)
```
1. Click "⚡ Benchmark Inference"
2. Show <1ms latency
3. Emphasize 600x speedup
4. Show reaction distance comparison
```

---

## 🔧 Customization

### Change Port
Edit `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8080)
```

### Use Different Model
Edit `app.py`:
```python
MODEL_PATH = './models/ppo_debris_avoidance/final_model.zip'
```

### Adjust Simulation Parameters
Edit `app.py` in `initialize_environment()`:
```python
env = OrbitalDebrisEnv(
    dt=10.0,              # Time step
    collision_radius=75.0, # Collision threshold
    initial_fuel=10.0,     # Fuel budget
    # ... etc
)
```

---

## 🐛 Troubleshooting

### "Model not loaded" warning
```bash
# Train a quick model first
python scripts/train.py --test
```

### Port 5000 already in use
```bash
# Kill process on Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Visualization not showing
- Clear browser cache (Ctrl+Shift+Del)
- Check browser console for errors (F12)
- Ensure Plotly CDN is accessible

---

## 📱 Mobile Friendly

The web app is responsive! Demo on:
- Laptops/Desktops
- Tablets
- Smartphones

---

## 🎬 Screen Recording Tips

### For Video Demos
1. Open browser in fullscreen (F11)
2. Use screen recorder (OBS, Windows Game Bar)
3. Record each feature separately:
   - Single scenario run
   - AI vs No-AI comparison
   - Benchmark results

### For Live Demos
1. Pre-load the page
2. Test all buttons work
3. Have backup screenshots ready
4. Zoom browser to 125% for visibility

---

## 🏆 Impressive Demo Script

**Opening:**
> "Let me show you our system in action. This is a live web interface where we can test real-time collision avoidance."

**Run Scenario:**
> "Here's a head-on collision course. Watch as our AI autonomously steers the satellite to safety... [click Run] ...There! No collision, minimal fuel used."

**Comparison:**
> "What if we didn't have AI? Let me show you... [click Compare] ...Without AI, the satellite collides. With AI, perfect avoidance."

**Benchmark:**
> "How fast is this? [click Benchmark] ...Under 1 millisecond per decision. That's 600 times faster than waiting for ground control!"

**Closing:**
> "This isn't just a simulation—this runs the actual neural network we'd deploy on satellites. 44 kilobytes, millisecond decisions, autonomous safety."

---

## 🌟 Pro Tips

1. **Pre-run everything** before demo to ensure it works
2. **Keep terminal visible** to show real-time processing
3. **Use "Head-On" scenario** for most dramatic visualization
4. **Point to metrics** as they update in real-time
5. **Explain LVLH frame** briefly for technical judges

---

## 📊 Metrics Checklist

Make sure to highlight:
- ✅ 0% collision rate (with AI)
- ✅ <1ms decision latency
- ✅ 15-30% fuel usage (efficient!)
- ✅ 600x faster than ground control
- ✅ 44KB model size (edge-ready)

---

**Ready to impress! 🚀 Your live web demo is now ready for the hackathon!**

---

## 🔗 Quick Links

- **Local Demo**: http://localhost:5000
- **GitHub README**: [README.md](README.md)
- **Training Guide**: [QUICKSTART.md](QUICKSTART.md)
- **Source Code**: `app.py`, `templates/`, `static/`

---

**Questions? The web interface includes an "About" section explaining everything!**
