"""Initial migration - base tables

Revision ID: 0001
Revises:
Create Date: 2025-10-08 01:20:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create TimescaleDB extension
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")

    # Create exchanges table
    op.create_table('exchanges',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('code', sa.String(length=20), nullable=False),
        sa.Column('api_base_url', sa.String(length=500), nullable=False),
        sa.Column('api_version', sa.String(length=20), nullable=True),
        sa.Column('is_testnet', sa.Boolean(), nullable=True),
        sa.Column('rate_limit_requests_per_minute', sa.Integer(), nullable=True),
        sa.Column('rate_limit_orders_per_second', sa.Integer(), nullable=True),
        sa.Column('rate_limit_weight_per_minute', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('last_heartbeat', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code'),
        sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_exchanges_name'), 'exchanges', ['name'], unique=False)

    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('api_key', sa.String(length=255), nullable=True),
        sa.Column('api_secret', sa.String(length=255), nullable=True),
        sa.Column('timezone', sa.String(length=50), nullable=True),
        sa.Column('language', sa.String(length=10), nullable=True),
        sa.Column('risk_level', sa.String(length=20), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_admin', sa.Boolean(), nullable=True),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('api_key'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=False)

    # Create trading_symbols table
    op.create_table('trading_symbols',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('base_asset', sa.String(length=10), nullable=False),
        sa.Column('quote_asset', sa.String(length=10), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('is_spot_trading', sa.Boolean(), nullable=True),
        sa.Column('is_margin_trading', sa.Boolean(), nullable=True),
        sa.Column('min_qty', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('max_qty', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('step_size', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('min_price', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('max_price', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('tick_size', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol')
    )
    op.create_index(op.f('ix_trading_symbols_symbol'), 'trading_symbols', ['symbol'], unique=False)

    # Create kline_data table (will be converted to hypertable)
    op.create_table('kline_data',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('symbol_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('exchange_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('open_price', postgresql.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('high_price', postgresql.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('low_price', postgresql.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('close_price', postgresql.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('volume', postgresql.DECIMAL(precision=20, scale=8), nullable=False),
        sa.Column('quote_volume', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('sma_20', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('ema_12', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('ema_26', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('rsi', postgresql.DECIMAL(precision=5, scale=2), nullable=True),
        sa.Column('macd', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('macd_signal', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('bollinger_upper', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('bollinger_lower', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['exchange_id'], ['exchanges.id'], ),
        sa.ForeignKeyConstraint(['symbol_id'], ['trading_symbols.id'], ),
        sa.PrimaryKeyConstraint('time', 'symbol_id', 'exchange_id')
    )

    # Create technical_analysis table
    op.create_table('technical_analysis',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('symbol_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('exchange_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('signal_type', sa.String(length=20), nullable=False),
        sa.Column('signal_strength', postgresql.DECIMAL(precision=3, scale=2), nullable=True),
        sa.Column('support_level', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('resistance_level', postgresql.DECIMAL(precision=20, scale=8), nullable=True),
        sa.Column('entry_conditions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('exit_conditions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('risk_factors', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('analysis_version', sa.String(length=20), nullable=True),
        sa.Column('confidence_score', postgresql.DECIMAL(precision=3, scale=2), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['exchange_id'], ['exchanges.id'], ),
        sa.ForeignKeyConstraint(['symbol_id'], ['trading_symbols.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Convert kline_data to TimescaleDB hypertable
    op.execute("SELECT create_hypertable('kline_data', 'time', chunk_time_interval => INTERVAL '1 day');")

    # Create indexes
    op.create_index('idx_exchanges_code', 'exchanges', ['code'], unique=False)
    op.create_index('idx_exchanges_status', 'exchanges', ['status'], unique=False)
    op.create_index('idx_kline_symbol_time', 'kline_data', ['symbol_id', 'time'], unique=False)
    op.create_index('idx_kline_exchange_time', 'kline_data', ['exchange_id', 'time'], unique=False)
    op.create_index('idx_technical_symbol_timeframe', 'technical_analysis', ['symbol_id', 'timeframe'], unique=False)
    op.create_index('idx_technical_signal_strength', 'technical_analysis', ['signal_strength'], unique=False)


def downgrade() -> None:
    # Drop tables
    op.drop_table('technical_analysis')
    op.drop_table('kline_data')
    op.drop_table('trading_symbols')
    op.drop_table('users')
    op.drop_table('exchanges')