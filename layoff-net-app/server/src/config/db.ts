import mongoose from 'mongoose';
import dotenv from 'dotenv';
dotenv.config();

const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/layoffnet';

export const connectDB = async (): Promise<void> => {
  try {
    await mongoose.connect(MONGO_URI);
    console.log('✅ MongoDB connected →', MONGO_URI);
  } catch (err) {
    console.error('❌ MongoDB error:', (err as Error).message);
    process.exit(1);
  }
};
