import { Request, Response } from 'express';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { Prediction } from '../models/prediction.model';

const ML_API = process.env.ML_API_URL || 'http://localhost:8000';

const buildRisk = (p: number): 'Low' | 'Medium' | 'High' =>
  p < 0.35 ? 'Low' : p < 0.60 ? 'Medium' : 'High';

// ── POST /api/predict ─────────────────────────────────────────
export const createPrediction = async (req: Request, res: Response): Promise<void> => {
  const {
    company, industry, country,
    funding_amount = 0, employee_count = 0,
    growth_rate, valuation = 0,
  } = req.body;

  if (!industry || !country || growth_rate === undefined) {
    res.status(400).json({ error: 'industry, country, growth_rate are required' });
    return;
  }

  const mlRes = await axios.post(`${ML_API}/predict`, {
    industry, country,
    funding_amount: Number(funding_amount),
    employee_count: Number(employee_count),
    growth_rate:    Number(growth_rate),
    valuation:      Number(valuation),
  });

  const result   = mlRes.data;
  const risk     = buildRisk(result.probability);

  const doc = await Prediction.create({
    uid: uuidv4(), company: company || 'Unknown',
    industry, country,
    funding_amount:    Number(funding_amount),
    employee_count:    Number(employee_count),
    growth_rate:       Number(growth_rate),
    valuation:         Number(valuation),
    layoff_probability: result.probability,
    layoff_prediction:  result.prediction,
    risk_level: risk,
  });

  res.status(201).json({ ...result, risk_level: risk, id: doc.uid });
};

// ── POST /api/predict/batch ───────────────────────────────────
export const batchPrediction = async (req: Request, res: Response): Promise<void> => {
  const { companies } = req.body;
  if (!Array.isArray(companies) || companies.length === 0) {
    res.status(400).json({ error: 'companies array required' });
    return;
  }
  const mlRes = await axios.post(`${ML_API}/predict/batch`, { companies });
  res.json(mlRes.data);
};

// ── GET /api/predictions ──────────────────────────────────────
export const getPredictions = async (req: Request, res: Response): Promise<void> => {
  const { page = '1', limit = '20', risk } = req.query as Record<string, string>;
  const filter = risk ? { risk_level: risk } : {};
  const docs   = await Prediction.find(filter)
    .sort({ created_at: -1 })
    .skip((+page - 1) * +limit)
    .limit(+limit);
  const total  = await Prediction.countDocuments(filter);
  res.json({ data: docs, total, page: +page, pages: Math.ceil(total / +limit) });
};

// ── GET /api/stats ────────────────────────────────────────────
export const getStats = async (_req: Request, res: Response): Promise<void> => {
  const [total, high, medium, low, avgAgg] = await Promise.all([
    Prediction.countDocuments(),
    Prediction.countDocuments({ risk_level: 'High' }),
    Prediction.countDocuments({ risk_level: 'Medium' }),
    Prediction.countDocuments({ risk_level: 'Low' }),
    Prediction.aggregate([{ $group: { _id: null, avg: { $avg: '$layoff_probability' } } }]),
  ]);
  res.json({ total, high, medium, low, avg_probability: avgAgg[0]?.avg?.toFixed(4) ?? 0 });
};

// ── GET /api/model/info ───────────────────────────────────────
export const getModelInfo = async (_req: Request, res: Response): Promise<void> => {
  const r = await axios.get(`${ML_API}/info`);
  res.json(r.data);
};
