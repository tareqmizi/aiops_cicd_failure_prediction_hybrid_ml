const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { Pool } = require('pg');
const { MongoClient } = require('mongodb');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;
const API_CORE_URL = process.env.API_CORE_URL || 'http://localhost:8000';

app.use(cors());
app.use(express.json());

// --- Database Connections & Fallbacks ---
let pgPool = null;
let mongoClient = null;
let mongoDb = null;

let useMockDb = false;

// Mock database storage for seamless fallback/testing
const mockDatabases = {
  repositories: [
    { name: 'ruby-amqp/amqp', build_count: 6096, fail_rate: 0.18 },
    { name: 'rails/rails', build_count: 1420, fail_rate: 0.22 },
    { name: 'travis-ci/travis-cookbooks', build_count: 512, fail_rate: 0.08 }
  ],
  commits: [],
  buildLogs: {},
  buildRcas: {}
};

async function initDatabases() {
  // 1. PostgreSQL Initialization
  try {
    pgPool = new Pool({
      connectionString: process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/aiops',
      connectionTimeoutMillis: 2000
    });
    // Verify connection and create tables
    await pgPool.query('SELECT NOW()');
    console.log('PostgreSQL connected successfully.');
    
    await pgPool.query(`
      CREATE TABLE IF NOT EXISTS repositories (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        build_count INT DEFAULT 0,
        fail_rate FLOAT DEFAULT 0.0
      );
      CREATE TABLE IF NOT EXISTS commits (
        id SERIAL PRIMARY KEY,
        repo_name VARCHAR(255) NOT NULL,
        git_branch VARCHAR(100),
        git_trigger_commit VARCHAR(100) UNIQUE,
        gh_build_started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        tr_status VARCHAR(50) DEFAULT 'passed',
        xgboost_probability FLOAT DEFAULT 0.0,
        lstm_probability FLOAT DEFAULT 0.0,
        ensemble_risk_score FLOAT DEFAULT 0.0,
        risk_category VARCHAR(50) DEFAULT 'Low'
      );
    `);
    console.log('PostgreSQL tables initialized.');
  } catch (err) {
    console.warn('PostgreSQL connection failed. Operating with In-Memory fallback database.', err.message);
    useMockDb = true;
  }

  // 2. MongoDB Initialization
  try {
    const mongoUri = process.env.MONGO_URL || 'mongodb://localhost:27017/aiops';
    mongoClient = new MongoClient(mongoUri, { serverSelectionTimeoutMS: 2000 });
    await mongoClient.connect();
    mongoDb = mongoClient.db();
    console.log('MongoDB connected successfully.');
  } catch (err) {
    console.warn('MongoDB connection failed. Operating with In-Memory fallback database.', err.message);
    useMockDb = true;
  }
}

// --- Endpoints ---

// Health Check
app.get('/health', async (req, res) => {
  let apiCoreHealthy = false;
  try {
    const r = await axios.get(`${API_CORE_URL}/health`);
    if (r.status === 200) apiCoreHealthy = true;
  } catch (e) {}

  res.json({
    status: 'healthy',
    mode: useMockDb ? 'in-memory-fallback' : 'database-connected',
    api_core_connected: apiCoreHealthy
  });
});

// GET /api/repositories
app.get('/api/repositories', async (req, res) => {
  if (useMockDb) {
    return res.json(mockDatabases.repositories);
  }

  try {
    const { rows } = await pgPool.query('SELECT * FROM repositories ORDER BY name');
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/commits
app.get('/api/commits', async (req, res) => {
  if (useMockDb) {
    return res.json(mockDatabases.commits.slice().sort((a, b) => new Date(b.gh_build_started_at) - new Date(a.gh_build_started_at)));
  }

  try {
    const { rows } = await pgPool.query('SELECT * FROM commits ORDER BY gh_build_started_at DESC');
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/builds/:commitHash/rca
app.get('/api/builds/:commitHash/rca', async (req, res) => {
  const { commitHash } = req.params;

  if (useMockDb) {
    const rca = mockDatabases.buildRcas[commitHash];
    const log = mockDatabases.buildLogs[commitHash] || 'No log found for this build.';
    if (!rca) {
      return res.status(404).json({ error: 'RCA records not found for this commit.' });
    }
    return res.json({ rca, log });
  }

  try {
    const rcaDoc = await mongoDb.collection('rca_results').findOne({ commitHash });
    const logDoc = await mongoDb.collection('build_logs').findOne({ commitHash });
    
    if (!rcaDoc) {
      return res.status(404).json({ error: 'RCA records not found for this commit.' });
    }
    res.json({
      rca: rcaDoc.rca,
      log: logDoc ? logDoc.log : 'No log found for this build.'
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/builds
app.post('/api/builds', async (req, res) => {
  const currentBuild = req.body;
  const repoName = currentBuild.gh_project_name || 'unknown-repo';
  const commitHash = currentBuild.git_trigger_commit || Math.random().toString(36).substring(7);

  try {
    // 1. Fetch history (last 10 builds) for sequence prediction
    let history = [];
    if (useMockDb) {
      history = mockDatabases.commits
        .filter(c => c.repo_name === repoName)
        .slice(-10);
    } else {
      const { rows } = await pgPool.query(
        'SELECT * FROM commits WHERE repo_name = $1 ORDER BY gh_build_started_at DESC LIMIT 10',
        [repoName]
      );
      history = rows.reverse();
    }

    // 2. Call FastAPI core for /predict
    let predictions = {
      xgboost_probability: 0.5,
      lstm_probability: 0.5,
      ensemble_risk_score: 0.5,
      risk_category: 'Medium'
    };

    try {
      const predRes = await axios.post(`${API_CORE_URL}/predict`, {
        current: currentBuild,
        history: history
      });
      predictions = predRes.data;
    } catch (e) {
      console.error('FastAPI Core prediction request failed. Defaulting to baseline.', e.message);
    }

    // 3. Save Commit Record
    const tr_status = currentBuild.tr_status || 'passed';
    let newRecord = {
      repo_name: repoName,
      git_branch: currentBuild.git_branch || 'main',
      git_trigger_commit: commitHash,
      gh_build_started_at: currentBuild.gh_build_started_at || new Date().toISOString(),
      tr_status,
      ...predictions
    };

    if (useMockDb) {
      // Check if commit already exists (prevent duplicate simulation)
      const existsIndex = mockDatabases.commits.findIndex(c => c.git_trigger_commit === commitHash);
      if (existsIndex > -1) {
        mockDatabases.commits[existsIndex] = { ...mockDatabases.commits[existsIndex], ...newRecord };
      } else {
        mockDatabases.commits.push(newRecord);
      }
      // Update Repo Stats
      let repo = mockDatabases.repositories.find(r => r.name === repoName);
      if (!repo) {
        repo = { name: repoName, build_count: 0, fail_rate: 0.0 };
        mockDatabases.repositories.push(repo);
      }
      repo.build_count += 1;
      const failedCount = mockDatabases.commits.filter(c => c.repo_name === repoName && ['failed', 'errored'].includes(c.tr_status)).length;
      repo.fail_rate = parseFloat((failedCount / repo.build_count).toFixed(4));
    } else {
      // Insert to Postgres
      await pgPool.query(
        `INSERT INTO commits (repo_name, git_branch, git_trigger_commit, gh_build_started_at, tr_status, xgboost_probability, lstm_probability, ensemble_risk_score, risk_category)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
         ON CONFLICT (git_trigger_commit) DO UPDATE SET
         tr_status = EXCLUDED.tr_status,
         xgboost_probability = EXCLUDED.xgboost_probability,
         lstm_probability = EXCLUDED.lstm_probability,
         ensemble_risk_score = EXCLUDED.ensemble_risk_score,
         risk_category = EXCLUDED.risk_category`,
        [
          newRecord.repo_name,
          newRecord.git_branch,
          newRecord.git_trigger_commit,
          newRecord.gh_build_started_at,
          newRecord.tr_status,
          newRecord.xgboost_probability,
          newRecord.lstm_probability,
          newRecord.ensemble_risk_score,
          newRecord.risk_category
        ]
      );

      // Upsert Repository record and update stats
      const totalBuildsRes = await pgPool.query('SELECT COUNT(*), COUNT(*) FILTER (WHERE tr_status IN (\'failed\', \'errored\')) as failed FROM commits WHERE repo_name = $1', [repoName]);
      const total = parseInt(totalBuildsRes.rows[0].count);
      const failed = parseInt(totalBuildsRes.rows[0].failed);
      const failRate = total > 0 ? failed / total : 0.0;

      await pgPool.query(
        `INSERT INTO repositories (name, build_count, fail_rate)
         VALUES ($1, $2, $3)
         ON CONFLICT (name) DO UPDATE SET
         build_count = EXCLUDED.build_count,
         fail_rate = EXCLUDED.fail_rate`,
        [repoName, total, failRate]
      );
    }

    // 4. Handle failed log & RCA triggers
    const logText = currentBuild.log_text || (['failed', 'errored'].includes(tr_status) ? 
      `[ERROR] Build step failed for commit ${commitHash} on branch ${newRecord.git_branch}.\n[ERROR] Process exited with code 1.` : null);

    let rcaAnalysis = null;
    if (logText || ['failed', 'errored'].includes(tr_status) || newRecord.ensemble_risk_score > 0.4) {
      const activeLog = logText || `[INFO] Simulating pipeline evaluation for commit ${commitHash}.\n[WARN] High risk commit detected (Risk Score: ${newRecord.ensemble_risk_score}).\n[ERROR] Potential dependency or integration conflict.`;
      
      // Save logs
      if (useMockDb) {
        mockDatabases.buildLogs[commitHash] = activeLog;
      } else {
        await mongoDb.collection('build_logs').replaceOne(
          { commitHash },
          { commitHash, log: activeLog },
          { upsert: true }
        );
      }

      // Fetch RCA analysis
      rcaAnalysis = {
        summary: 'Risk Threshold Exceeded / Build Issue',
        root_cause: 'Predictive models flagged elevated probability of CI pipeline failure.',
        recommendations: ['Perform code review on high churn files.', 'Run integration test suite before merging.'],
        source: 'orchestrator-fallback'
      };

      try {
        const rcaRes = await axios.post(`${API_CORE_URL}/rca`, { log_text: activeLog });
        rcaAnalysis = rcaRes.data;
      } catch (err) {
        console.error('FastAPI Core RCA log analysis failed.', err.message);
      }

      // Save RCA result
      if (useMockDb) {
        mockDatabases.buildRcas[commitHash] = rcaAnalysis;
      } else {
        await mongoDb.collection('rca_results').replaceOne(
          { commitHash },
          { commitHash, rca: rcaAnalysis },
          { upsert: true }
        );
      }
    }

    res.status(201).json({ ...newRecord, rca: rcaAnalysis, log: mockDatabases.buildLogs[commitHash] });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Helper function to process webhooks and batch items
async function processCommitWithAi(commitData, provider = 'Webhook') {
  const repoName = commitData.gh_project_name || commitData.repo_name || 'sample-repo';
  const commitHash = commitData.git_trigger_commit || commitData.commit_hash || Math.random().toString(36).substring(7);
  const branch = commitData.git_branch || 'main';
  const status = commitData.tr_status || 'passed';

  // 1. Fetch history
  let history = [];
  if (useMockDb) {
    history = mockDatabases.commits.filter(c => c.repo_name === repoName).slice(-10);
  } else {
    try {
      const { rows } = await pgPool.query(
        'SELECT * FROM commits WHERE repo_name = $1 ORDER BY gh_build_started_at DESC LIMIT 10',
        [repoName]
      );
      history = rows.reverse();
    } catch (e) {}
  }

  // 2. Call FastAPI Predict
  let predictions = { xgboost_probability: 0.25, lstm_probability: 0.3, ensemble_risk_score: 0.275, risk_category: 'Low' };
  try {
    const predRes = await axios.post(`${API_CORE_URL}/predict`, { current: commitData, history });
    predictions = predRes.data;
  } catch (e) {
    console.error('FastAPI Predict failed during webhook:', e.message);
  }

  // 3. Form record
  const newRecord = {
    provider,
    repo_name: repoName,
    git_branch: branch,
    git_trigger_commit: commitHash,
    gh_build_started_at: commitData.gh_build_started_at || new Date().toISOString(),
    tr_status: status,
    commit_message: commitData.commit_message || 'Merged commit or push update',
    author: commitData.author || 'Developer',
    ...predictions
  };

  // Save to memory / DB
  if (useMockDb) {
    const idx = mockDatabases.commits.findIndex(c => c.git_trigger_commit === commitHash);
    if (idx > -1) mockDatabases.commits[idx] = newRecord;
    else mockDatabases.commits.push(newRecord);
  } else {
    try {
      await pgPool.query(
        `INSERT INTO commits (repo_name, git_branch, git_trigger_commit, gh_build_started_at, tr_status, xgboost_probability, lstm_probability, ensemble_risk_score, risk_category)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
         ON CONFLICT (git_trigger_commit) DO UPDATE SET
         tr_status = EXCLUDED.tr_status, xgboost_probability = EXCLUDED.xgboost_probability,
         lstm_probability = EXCLUDED.lstm_probability, ensemble_risk_score = EXCLUDED.ensemble_risk_score, risk_category = EXCLUDED.risk_category`,
        [newRecord.repo_name, newRecord.git_branch, newRecord.git_trigger_commit, newRecord.gh_build_started_at, newRecord.tr_status, newRecord.xgboost_probability, newRecord.lstm_probability, newRecord.ensemble_risk_score, newRecord.risk_category]
      );
    } catch (e) {}
  }

  // 4. Trigger LLM Reasoning
  const logText = commitData.log_text || `[INFO] REST API triggered prediction for commit ${commitHash} on ${repoName}.\n[INFO] Branch: ${branch}, Status: ${status}.\n[ERROR] Pipeline verification flagged risk probability of ${(predictions.ensemble_risk_score * 100).toFixed(1)}%.`;
  
  let rcaAnalysis = {
    summary: predictions.risk_category === 'High' ? 'High Build Failure Risk Detected' : 'Pipeline Execution Stable',
    root_cause: `Ensemble Model (XGBoost + BiLSTM) evaluated change pattern with score ${(predictions.ensemble_risk_score * 100).toFixed(1)}%.`,
    recommendations: ['Review diff churn before deployment.', 'Ensure unit & integration tests pass.'],
    source: 'aiops-llm-reasoner'
  };

  try {
    const rcaRes = await axios.post(`${API_CORE_URL}/rca`, { log_text: logText });
    rcaAnalysis = rcaRes.data;
  } catch (err) {}

  if (useMockDb) {
    mockDatabases.buildLogs[commitHash] = logText;
    mockDatabases.buildRcas[commitHash] = rcaAnalysis;
  } else {
    try {
      await mongoDb.collection('build_logs').replaceOne({ commitHash }, { commitHash, log: logText }, { upsert: true });
      await mongoDb.collection('rca_results').replaceOne({ commitHash }, { commitHash, rca: rcaAnalysis }, { upsert: true });
    } catch (e) {}
  }

  return { ...newRecord, rca: rcaAnalysis, log: logText };
}

// REST Single Commit Prediction Endpoint (On-demand single commit evaluation, no webhooks)
app.post('/api/predict-commit', async (req, res) => {
  try {
    const commitData = req.body;
    const result = await processCommitWithAi(commitData, 'REST-API');
    res.status(200).json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Batch Predict Endpoint for Uploaded CSV rows
app.post('/api/batch-predict', async (req, res) => {
  try {
    const { items } = req.body;
    if (!Array.isArray(items) || items.length === 0) {
      return res.status(400).json({ error: 'Payload must contain a non-empty "items" array.' });
    }

    const processedResults = [];
    for (const item of items.slice(0, 50)) {
      const result = await processCommitWithAi(item, 'CSV Upload');
      processedResults.push(result);
    }

    res.status(200).json({ status: 'success', count: processedResults.length, results: processedResults });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Start Express Server
initDatabases().then(() => {
  app.listen(PORT, () => {
    console.log(`Backend Orchestrator running on port ${PORT}`);
  });
});
