require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const mongoose = require('mongoose');
const rateLimit = require('express-rate-limit');
const { v4: uuidv4 } = require('uuid');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 5000;
const ML_API = process.env.ML_API_URL || 'http://localhost:8000';

// ── Middleware ────────────────────────────────────────────────
app.use(helmet());
app.use(cors({ origin: process.env.CLIENT_URL || 'http://localhost:3000' }));
app.use(express.json());
app.use(morgan('dev'));
app.use(rateLimit({ windowMs: 15 * 60 * 1000, max: 200 }));

// ── MongoDB ───────────────────────────────────────────────────
mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/layoffnet', {
  useNewUrlParser: true, useUnifiedTopology: true
}).then(() => console.log('✓ MongoDB connected'))
  .catch(err => console.log('MongoDB optional:', err.message));

// ── Schemas ───────────────────────────────────────────────────
const predictionSchema = new mongoose.Schema({
  id: { type: String, default: uuidv4 },
  company: String,
  industry: String,
  country: String,
  funding_amount: Number,
  employee_count: Number,
  growth_rate: Number,
  valuation: Number,
  layoff_probability: Number,
  layoff_prediction: Number,
  risk_level: String,
  created_at: { type: Date, default: Date.now }
});
const Prediction = mongoose.model('Prediction', predictionSchema);

// ── Routes ────────────────────────────────────────────────────

// Health
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', service: 'LAYOFF-NET API', version: '1.0.0',
             timestamp: new Date().toISOString() });
});

// Predict single company
app.post('/api/predict', async (req, res) => {
  try {
    const { company, industry, country, funding_amount,
            employee_count, growth_rate, valuation } = req.body;

    if (!industry || !country || growth_rate === undefined) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const mlRes = await axios.post(`${ML_API}/predict`, {
      industry, country,
      funding_amount: Number(funding_amount) || 0,
      employee_count: Number(employee_count) || 0,
      growth_rate: Number(growth_rate),
      valuation: Number(valuation) || 0
    });

    const result = mlRes.data;
    const risk = result.probability < 0.35 ? 'Low'
               : result.probability < 0.60 ? 'Medium' : 'High';

    const doc = new Prediction({
      id: uuidv4(), company: company || 'Unknown',
      industry, country,
      funding_amount: Number(funding_amount) || 0,
      employee_count: Number(employee_count) || 0,
      growth_rate: Number(growth_rate),
      valuation: Number(valuation) || 0,
      layoff_probability: result.probability,
      layoff_prediction: result.prediction,
      risk_level: risk
    });
    await doc.save().catch(() => {});

    res.json({ ...result, risk_level: risk, id: doc.id });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Predict batch
app.post('/api/predict/batch', async (req, res) => {
  try {
    const { companies } = req.body;
    if (!Array.isArray(companies) || companies.length === 0)
      return res.status(400).json({ error: 'companies array required' });

    const mlRes = await axios.post(`${ML_API}/predict/batch`, { companies });
    res.json(mlRes.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get history
app.get('/api/predictions', async (req, res) => {
  try {
    const { page = 1, limit = 20, risk } = req.query;
    const filter = risk ? { risk_level: risk } : {};
    const docs = await Prediction.find(filter)
      .sort({ created_at: -1 })
      .skip((page - 1) * limit)
      .limit(Number(limit));
    const total = await Prediction.countDocuments(filter);
    res.json({ data: docs, total, page: Number(page), pages: Math.ceil(total / limit) });
  } catch (err) {
    res.status(500).json({ data: [], total: 0, error: err.message });
  }
});

// Stats
app.get('/api/stats', async (req, res) => {
  try {
    const total = await Prediction.countDocuments();
    const high   = await Prediction.countDocuments({ risk_level: 'High' });
    const medium = await Prediction.countDocuments({ risk_level: 'Medium' });
    const low    = await Prediction.countDocuments({ risk_level: 'Low' });
    const avgProb = await Prediction.aggregate([
      { $group: { _id: null, avg: { $avg: '$layoff_probability' } } }
    ]);
    res.json({
      total, high, medium, low,
      avg_probability: avgProb[0]?.avg?.toFixed(4) || 0
    });
  } catch {
    res.json({ total: 0, high: 0, medium: 0, low: 0, avg_probability: 0 });
  }
});

// Model info from ML API
app.get('/api/model/info', async (req, res) => {
  try {
    const r = await axios.get(`${ML_API}/info`);
    res.json(r.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── Start ─────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`✓ LAYOFF-NET Server → http://localhost:${PORT}`);
  console.log(`  ML API target    → ${ML_API}`);
});
