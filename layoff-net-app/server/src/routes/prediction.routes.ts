import { Router, Request, Response } from 'express';
import {
  createPrediction,
  batchPrediction,
  getPredictions,
  getStats,
  getModelInfo,
} from '../controllers/prediction.controller';
import { asyncHandler } from '../utils/asyncHandler';

const router = Router();

router.post('/predict',         asyncHandler(createPrediction));
router.post('/predict/batch',   asyncHandler(batchPrediction));
router.get('/predictions',      asyncHandler(getPredictions));
router.get('/stats',            asyncHandler(getStats));
router.get('/model/info',       asyncHandler(getModelInfo));

export default router;
