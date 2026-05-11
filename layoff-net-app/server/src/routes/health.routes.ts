import { Router, Request, Response } from 'express';

const router = Router();

router.get('/', (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    service: 'LAYOFF-NET API',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    endpoints: {
      predict:       'POST /api/predict',
      batch:         'POST /api/predict/batch',
      predictions:   'GET  /api/predictions',
      stats:         'GET  /api/stats',
      model_info:    'GET  /api/model/info',
    }
  });
});

export default router;
