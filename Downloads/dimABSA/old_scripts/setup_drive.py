from pydrive.auth import GoogleAuth

def login_and_save():
    gauth = GoogleAuth()
    
    # 1. Try to load existing credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    
    if gauth.credentials is None:
        # No creds found, ask for authentication
        print("--- ⚠️  Action Required: Click the link below to authenticate ---")
        gauth.CommandLineAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # We are good
        gauth.Authorize()
        
    # 2. Save the credentials so we don't have to log in again later
    gauth.SaveCredentialsFile("mycreds.txt")
    print("✅ Credentials saved to 'mycreds.txt'. You are ready to run the benchmark!")

if __name__ == "__main__":
    login_and_save()