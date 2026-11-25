
from dotenv import load_dotenv
from app.core.supabase_client import get_supabase_client

load_dotenv()

supabase = get_supabase_client()

result = (
    supabase
    .table("conversation_sessions")
    .select("id")
    .limit(1)
    .execute()
)

print("Supabase connection successful!")
print("Database query executed successfully.")
