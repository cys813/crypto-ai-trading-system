-- Initialize TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Set default timezone
SET timezone = 'UTC';

-- Create initial indexes for better performance
-- These will be created after the tables are created by Alembic