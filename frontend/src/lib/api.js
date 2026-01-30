import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
export const API = axios.create({
  baseURL: `${BACKEND_URL}/api`,
  timeout: 30000,
});

// Market Data
export const getMarketData = async (symbol, timeframe = '5m', limit = 100) => {
  const response = await API.post('/market/data', { symbol, timeframe, limit });
  return response.data;
};

export const getLatestPrice = async (symbol) => {
  const response = await API.get(`/market/latest/${symbol.replace('/', '-')}`);
  return response.data;
};

export const getSymbols = async () => {
  const response = await API.get('/market/symbols');
  return response.data;
};

// Predictions
export const makePrediction = async (symbol, useSentiment = true) => {
  const response = await API.post('/predict', { symbol, use_sentiment: useSentiment });
  return response.data;
};

export const getPredictionHistory = async (symbol = null, limit = 50) => {
  const params = new URLSearchParams();
  if (symbol) params.append('symbol', symbol);
  params.append('limit', limit);
  const response = await API.get(`/predictions/history?${params.toString()}`);
  return response.data;
};

// Sentiment
export const analyzeSentiment = async (text, useLlm = false) => {
  const response = await API.post('/sentiment/analyze', { text, use_llm: useLlm });
  return response.data;
};

export const getAggregateSentiment = async (symbol) => {
  const response = await API.get(`/sentiment/aggregate/${symbol.replace('/', '-')}`);
  return response.data;
};

// Training
export const startTraining = async (symbol, epochs = 50, batchSize = 32, lookbackDays = 30) => {
  const response = await API.post('/training/start', {
    symbol,
    epochs,
    batch_size: batchSize,
    lookback_days: lookbackDays
  });
  return response.data;
};

export const getTrainingStatus = async () => {
  const response = await API.get('/training/status');
  return response.data;
};

export const stopTraining = async () => {
  const response = await API.post('/training/stop');
  return response.data;
};

export const getTrainingHistory = async (limit = 10) => {
  const response = await API.get(`/training/history?limit=${limit}`);
  return response.data;
};

// Model
export const getModelSummary = async () => {
  const response = await API.get('/model/summary');
  return response.data;
};

export const buildModel = async (config = null) => {
  const response = await API.post('/model/build', config);
  return response.data;
};

// Dashboard
export const getDashboardStats = async () => {
  const response = await API.get('/dashboard/stats');
  return response.data;
};

// Health
export const healthCheck = async () => {
  const response = await API.get('/health');
  return response.data;
};
