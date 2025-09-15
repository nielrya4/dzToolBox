"""Add database indices for performance (simple)

Revision ID: 22a307b8a94b
Revises: 7f42d8fcab0c
Create Date: 2025-09-12 09:19:36.126630

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '22a307b8a94b'
down_revision = '7f42d8fcab0c'
branch_labels = None
depends_on = None


def upgrade():
    # Add new columns to code_file table only if they don't exist
    try:
        op.add_column('code_file', sa.Column('created_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=True))
    except:
        pass  # Column may already exist
    
    try:
        op.add_column('code_file', sa.Column('updated_at', sa.DateTime(), server_default=sa.func.current_timestamp(), nullable=True))
    except:
        pass  # Column may already exist
    
    # Create composite indexes for better query performance
    try:
        op.create_index('idx_user_created', 'code_file', ['user_id', 'created_at'])
    except:
        pass  # Index may already exist
        
    try:
        op.create_index('idx_user_updated', 'code_file', ['user_id', 'updated_at'])
    except:
        pass  # Index may already exist
        
    try:
        op.create_index('ix_code_file_created_at', 'code_file', ['created_at'])
    except:
        pass  # Index may already exist


def downgrade():
    # Remove indexes if they exist
    try:
        op.drop_index('ix_code_file_created_at', table_name='code_file')
    except:
        pass
        
    try:
        op.drop_index('idx_user_updated', table_name='code_file')
    except:
        pass
        
    try:
        op.drop_index('idx_user_created', table_name='code_file')
    except:
        pass
    
    # Remove columns if they exist
    try:
        op.drop_column('code_file', 'updated_at')
    except:
        pass
        
    try:
        op.drop_column('code_file', 'created_at')
    except:
        pass