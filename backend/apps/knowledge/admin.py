from django.contrib import admin

from .models import KnowledgeChunk, KnowledgeDocument

admin.site.register(KnowledgeDocument)
admin.site.register(KnowledgeChunk)
