from django.contrib.auth import get_user_model, password_validation
from django.core.exceptions import ValidationError
from rest_framework import serializers

from .models import Conversation, Memory, Message, Profile


class CredentialsSerializer(serializers.Serializer):
    email = serializers.EmailField(max_length=150)
    password = serializers.CharField(max_length=128, trim_whitespace=False, write_only=True)

    def validate_email(self, value):
        return value.lower()


class RegistrationSerializer(CredentialsSerializer):
    name = serializers.CharField(max_length=150)

    def validate(self, attrs):
        user = get_user_model()(username=attrs['email'], email=attrs['email'], first_name=attrs['name'])
        try:
            password_validation.validate_password(attrs['password'], user)
        except ValidationError as exc:
            raise serializers.ValidationError({'password': exc.messages}) from exc
        return attrs


class AccountSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=150, required=False)
    memory_enabled = serializers.BooleanField(required=False)


def account_data(user):
    profile, _ = Profile.objects.get_or_create(user=user)
    return {
        'id': user.pk, 'name': user.first_name or user.email,
        'email': user.email, 'memory_enabled': profile.memory_enabled,
    }


class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = ['id', 'title', 'language', 'created_at', 'updated_at']
        read_only_fields = ['id', 'language', 'created_at', 'updated_at']


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'role', 'content', 'response', 'created_at']


class MemorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Memory
        fields = ['id', 'content', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']
