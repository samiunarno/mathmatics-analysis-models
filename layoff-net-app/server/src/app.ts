import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import predictionRouter from './routes/prediction.routes';
import healthRouter     from './routes/health.routes';

dotenv.config();

const app = express();

// ── Security & Middleware ─────────────────────────────────────
app.use(helmet());
app.use(cors({ origin: process.env.CLIENT_URL || 'http://localhost:3000' }));
app.use(express.json());
app.use(morgan('dev'));
app.use(rateLimit({ windowMs: 15 * 60 * 1000, max: 200 }));

// ── Routes ────────────────────────────────────────────────────
app.use('/api/health', healthRouter);
app.use('/api',        predictionRouter);

// ── Global Error Handler ──────────────────────────────────────
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error('❌', err.message);
  res.status(500).json({ error: err.message || 'Internal Server Error' });
});

export default app;
