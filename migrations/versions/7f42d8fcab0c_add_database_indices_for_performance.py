"""add database indices for performance

Revision ID: 7f42d8fcab0c
Revises: 
Create Date: 2025-08-26 15:16:02.531509

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7f42d8fcab0c'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Add indices only - don't modify existing columns
    with op.batch_alter_table('code_file', schema=None) as batch_op:
        batch_op.create_index('idx_user_title', ['user_id', 'title'], unique=False)
        batch_op.create_index(batch_op.f('ix_code_file_title'), ['title'], unique=False)
        batch_op.create_index(batch_op.f('ix_code_file_user_id'), ['user_id'], unique=False)

    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_user_username'), ['username'], unique=False)


def downgrade():
    # Drop indices in reverse order
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_user_username'))

    with op.batch_alter_table('code_file', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_code_file_user_id'))
        batch_op.drop_index(batch_op.f('ix_code_file_title'))
        batch_op.drop_index('idx_user_title')