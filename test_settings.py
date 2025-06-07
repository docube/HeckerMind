from app.config.settings import get_settings

settings = get_settings()

print(settings.OPENAI_API_KEY)
print(settings.DEBUG)
