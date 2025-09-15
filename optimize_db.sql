-- Database Performance Optimizations for dzToolBox
-- Add timestamp columns and performance indexes

-- Add timestamp columns if they don't exist
ALTER TABLE code_file 
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

ALTER TABLE code_file 
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_user_created ON code_file(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_user_updated ON code_file(user_id, updated_at);
CREATE INDEX IF NOT EXISTS ix_code_file_created_at ON code_file(created_at);

-- Improve existing indexes
CREATE INDEX IF NOT EXISTS idx_code_file_user_id ON code_file(user_id);
CREATE INDEX IF NOT EXISTS idx_code_file_title ON code_file(title);

-- Add username index for user lookups
CREATE INDEX IF NOT EXISTS idx_user_username ON "user"(username);

-- Update foreign key constraint to CASCADE for better cleanup performance
-- First check if constraint exists and drop it
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 
        FROM information_schema.table_constraints 
        WHERE constraint_name = 'code_file_user_id_fkey' 
        AND table_name = 'code_file'
    ) THEN
        ALTER TABLE code_file DROP CONSTRAINT code_file_user_id_fkey;
    END IF;
END $$;

-- Add the new constraint with CASCADE
ALTER TABLE code_file 
ADD CONSTRAINT code_file_user_id_fkey 
FOREIGN KEY (user_id) REFERENCES "user"(id) ON DELETE CASCADE;

-- Set default values for existing rows
UPDATE code_file 
SET created_at = CURRENT_TIMESTAMP 
WHERE created_at IS NULL;

UPDATE code_file 
SET updated_at = CURRENT_TIMESTAMP 
WHERE updated_at IS NULL;

-- Analyze tables for better query planning
ANALYZE code_file;
ANALYZE "user";

-- Display optimization results
SELECT 'Database optimizations completed successfully' as status;