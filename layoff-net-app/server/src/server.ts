import dotenv from 'dotenv';
dotenv.config();

import { connectDB } from './config/db';
import app from './app';

const PORT = Number(process.env.PORT) || 5001;

(async () => {
  await connectDB();
  app.listen(PORT, () => {
    console.log('');
    console.log('╔═══════════════════════════════════════════╗');
    console.log('║      LAYOFF-NET  Backend  Running 🚀       ║');
    console.log(`║   → http://localhost:${PORT}/api/health      ║`);
    console.log('╚═══════════════════════════════════════════╝');
  });
})();
