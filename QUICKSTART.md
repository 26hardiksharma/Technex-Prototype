# 🚀 Quick Start Guide

Get your autonomous debris avoidance system running in 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd d:\TnX

# Install all requirements
pip install -r requirements.txt
```

**Expected output:** All packages install successfully

## Step 2: Test Installation (30 seconds)

```bash
# Run quick system test
python demo.py --dynamics
```

**Expected output:** 
```
✓ Orbital period: 90.0 minutes
✓ Initial position: [100.0, -5000.0] m
✓ Final position: [99.9, -4500.3] m
✅ Orbital dynamics working!
```

## Step 3: Full Demo (3 minutes)

```bash
# Run complete system demo
python demo.py --full
```

This will:
- ✅ Test orbital dynamics
- ✅ Test environment
- ✅ Train tiny agent (10K steps)
- ✅ Evaluate performance
- ✅ Benchmark inference time
- ✅ Export to ONNX

## Step 4: Production Training (4-6 hours)

```bash
# Train full agent
python scripts/train.py --timesteps 500000 --envs 4

# Monitor in another terminal
tensorboard --logdir logs
```

Open browser to `http://localhost:6006` to watch training live!

## Step 5: Evaluate & Showcase

```bash
# Evaluate trained agent
python scripts/evaluate.py \
    --model models/ppo_debris_avoidance/final_model.zip \
    --episodes 50 \
    --output results/

# Benchmark edge performance
python scripts/benchmark_inference.py \
    --model models/ppo_debris_avoidance/final_model.zip \
    --samples 1000
```

---

## 🎯 For Hackathon Demo

### Quick Training Demo (2 minutes)
```bash
python scripts/train.py --test
```

### Show Results
```bash
# 1. Visualize trajectory
python scripts/evaluate.py \
    --model models/test_run/final_model.zip \
    --episodes 3

# 2. Show edge performance
python scripts/benchmark_inference.py \
    --model models/test_run/final_model.zip \
    --samples 100
```

### Key Talking Points

1. **Problem**: Space debris threatens satellites
2. **Solution**: Autonomous onboard AI
3. **Innovation**: <1ms decisions vs 600ms ground control
4. **Impact**: 600x faster = better safety margins
5. **Deployment**: Tiny model (40KB) runs on satellite hardware

---

## 📊 Expected Results

After full training, you should see:

| Metric | Target | Typical Result |
|--------|--------|----------------|
| Success Rate | >90% | 95%+ |
| Collision Rate | <10% | <5% |
| Fuel Usage | <50% | ~30% |
| Inference Time | <10ms | <1ms |

---

## 🐛 Troubleshooting

### "Module not found"
```bash
# Add to Python path
set PYTHONPATH=%PYTHONPATH%;d:\TnX
```

### "CUDA not available"
Don't worry! CPU training works fine:
```bash
python scripts/train.py --timesteps 100000 --envs 2
```

### "Out of memory"
Reduce parallel environments:
```bash
python scripts/train.py --envs 2
```

---

## 🎨 Visualizations

All plots saved to `results/`:
- `best_random.png` - Best avoidance trajectory
- `scenario_comparison.png` - Performance across scenarios
- `evaluation_report.txt` - Detailed metrics

---

## 📝 Files You'll Show in Demo

1. **README.md** - Overview and problem statement
2. **demo.py** - Live demonstration
3. **results/*.png** - Trajectory visualizations
4. **logs/** - TensorBoard training curves
5. **models/*.onnx** - Deployed model

---

## 🏆 Presentation Tips

1. **Hook**: Start with Kessler Syndrome threat
2. **Problem**: Show 600ms ground control delay
3. **Demo**: Run live collision avoidance
4. **Results**: Show <1ms inference time
5. **Impact**: 600x speedup = lives saved

**Opening line**: 
> "What if satellites could dodge debris in 1 millisecond instead of waiting 600 milliseconds for ground control? That's the difference between collision and survival."

---

## ⚡ Speed Run (Complete in 10 minutes)

```bash
# 1. Install (2 min)
pip install -r requirements.txt

# 2. Quick demo (3 min)
python demo.py --full

# 3. Show results (5 min)
python scripts/evaluate.py --model models/demo_run/final_model.zip --episodes 5
python scripts/benchmark_inference.py --model models/demo_run/final_model.zip --samples 100
```

Done! You now have a working prototype.

---

**Need help?** Check the full README.md for detailed documentation.

**Ready for hackathon?** Run `python demo.py --full` and you're good to go! 🚀
