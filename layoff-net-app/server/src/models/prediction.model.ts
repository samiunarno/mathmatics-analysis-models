import { Schema, model, Document } from 'mongoose';
import { v4 as uuidv4 } from 'uuid';

export interface IPrediction extends Document {
  uid: string;
  company: string;
  industry: string;
  country: string;
  funding_amount: number;
  employee_count: number;
  growth_rate: number;
  valuation: number;
  layoff_probability: number;
  layoff_prediction: number;
  risk_level: 'Low' | 'Medium' | 'High';
  created_at: Date;
}

const predictionSchema = new Schema<IPrediction>({
  uid:               { type: String, default: uuidv4 },
  company:           { type: String, default: 'Unknown' },
  industry:          { type: String, required: true },
  country:           { type: String, required: true },
  funding_amount:    { type: Number, default: 0 },
  employee_count:    { type: Number, default: 0 },
  growth_rate:       { type: Number, required: true },
  valuation:         { type: Number, default: 0 },
  layoff_probability:{ type: Number, required: true },
  layoff_prediction: { type: Number, required: true },
  risk_level:        { type: String, enum: ['Low','Medium','High'], required: true },
  created_at:        { type: Date, default: Date.now },
});

export const Prediction = model<IPrediction>('Prediction', predictionSchema);
