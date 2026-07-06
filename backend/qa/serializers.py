"""
DRF serializers for the Islamic RAG query API.

Matches the data models defined in PRD §8.
"""

from rest_framework import serializers


class QueryRequestSerializer(serializers.Serializer):
    query = serializers.CharField(required=True, min_length=1, max_length=2000)
    language = serializers.ChoiceField(
        choices=['ar', 'en', 'id'],
        default='en',
    )
    max_sources = serializers.IntegerField(default=5, min_value=1, max_value=20)
    include_arabic = serializers.BooleanField(default=True)


class SourceSerializer(serializers.Serializer):
    source_tag = serializers.CharField()
    corpus = serializers.ChoiceField(choices=['quran', 'hadith'])
    text_ar = serializers.CharField(allow_blank=True)
    text_en = serializers.CharField(allow_blank=True)
    verification_status = serializers.ChoiceField(
        choices=['exact', 'normalized', 'fuzzy', 'semantic', 'hallucinated', 'unknown']
    )
    retrieval_score = serializers.FloatField()


class SafetyResultSerializer(serializers.Serializer):
    hallucination_detected = serializers.BooleanField(default=False)
    flagged_spans = serializers.ListField(child=serializers.CharField(), default=list)
    fatwa_boundary_triggered = serializers.BooleanField(default=False)
    disclaimer = serializers.CharField(allow_null=True, default=None)


class QueryResponseSerializer(serializers.Serializer):
    query = serializers.CharField()
    intent = serializers.CharField(default='general')
    answer = serializers.CharField()
    sources = SourceSerializer(many=True)
    citations = serializers.ListField(child=serializers.CharField())
    safety = SafetyResultSerializer()
    pipeline_meta = serializers.DictField(child=serializers.CharField())