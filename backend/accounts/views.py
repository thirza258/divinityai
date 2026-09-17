from django.contrib.auth import authenticate, get_user_model, login, logout
from django.db import IntegrityError, transaction
from django.db.models import Q
from django.middleware.csrf import get_token
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_protect
from rest_framework.authentication import SessionAuthentication
from rest_framework.exceptions import ValidationError
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import SimpleRateThrottle
from rest_framework.views import APIView

from .models import Conversation, Memory, Profile
from .serializers import (
    AccountSerializer, ConversationSerializer, CredentialsSerializer,
    MemorySerializer, MessageSerializer, RegistrationSerializer, account_data,
)


class AuthThrottle(SimpleRateThrottle):
    scope = 'account_auth'
    rate = '10/min'

    def get_cache_key(self, request, view):
        return self.cache_format % {'scope': self.scope, 'ident': request.META.get('REMOTE_ADDR', '')}


@method_decorator(never_cache, name='dispatch')
class AccountAPIView(APIView):
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAuthenticated]


class SessionView(AccountAPIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({
            'user': account_data(request.user) if request.user.is_authenticated else None,
            'csrf_token': get_token(request),
        })


# DRF only enforces CSRF for authenticated sessions. Login and registration
# must also enforce it before an anonymous visitor becomes authenticated.
@method_decorator(csrf_protect, name='dispatch')
class RegisterView(AccountAPIView):
    permission_classes = [AllowAny]
    throttle_classes = [AuthThrottle]

    def post(self, request):
        serializer = RegistrationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        try:
            with transaction.atomic():
                user = get_user_model().objects.create_user(
                    username=data['email'], email=data['email'],
                    first_name=data['name'], password=data['password'],
                )
                Profile.objects.create(user=user)
        except IntegrityError:
            return Response({'detail': 'Unable to create an account with this email. Try signing in.'}, status=400)
        login(request, user)
        return Response({'user': account_data(user), 'csrf_token': get_token(request)}, status=201)


@method_decorator(csrf_protect, name='dispatch')
class LoginView(AccountAPIView):
    permission_classes = [AllowAny]
    throttle_classes = [AuthThrottle]

    def post(self, request):
        serializer = CredentialsSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        user = authenticate(request, username=data['email'], password=data['password'])
        if user is None:
            return Response({'detail': 'Email or password is incorrect.'}, status=400)
        login(request, user)
        return Response({'user': account_data(user), 'csrf_token': get_token(request)})


class LogoutView(AccountAPIView):
    def post(self, request):
        logout(request)
        return Response({'user': None, 'csrf_token': get_token(request)})


class ProfileView(AccountAPIView):
    def patch(self, request):
        serializer = AccountSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        with transaction.atomic():
            if 'name' in data:
                request.user.first_name = data['name']
                request.user.save(update_fields=['first_name'])
            if 'memory_enabled' in data:
                Profile.objects.update_or_create(user=request.user, defaults={'memory_enabled': data['memory_enabled']})
        return Response({'user': account_data(request.user)})


class HistoryPagination(PageNumberPagination):
    page_size = 30


class ConversationListView(AccountAPIView):
    def get(self, request):
        conversations = Conversation.objects.filter(user=request.user)
        search = request.query_params.get('search', '').strip()[:200]
        if search:
            conversations = conversations.filter(
                Q(title__icontains=search) | Q(messages__content__icontains=search)
            ).distinct()
        paginator = HistoryPagination()
        page = paginator.paginate_queryset(conversations, request)
        return paginator.get_paginated_response(ConversationSerializer(page, many=True).data)

    def delete(self, request):
        Conversation.objects.filter(user=request.user).delete()
        return Response(status=204)


class ConversationDetailView(AccountAPIView):
    def get_conversation(self, request, pk):
        return get_object_or_404(Conversation, pk=pk, user=request.user)

    def get(self, request, pk):
        conversation = self.get_conversation(request, pk)
        return Response({
            **ConversationSerializer(conversation).data,
            'messages': MessageSerializer(conversation.messages.all(), many=True).data,
        })

    def patch(self, request, pk):
        serializer = ConversationSerializer(self.get_conversation(request, pk), data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    def delete(self, request, pk):
        self.get_conversation(request, pk).delete()
        return Response(status=204)


class MemoryListView(AccountAPIView):
    def get(self, request):
        return Response(MemorySerializer(Memory.objects.filter(user=request.user), many=True).data)

    def post(self, request):
        serializer = MemorySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        with transaction.atomic():
            # Serialize changes per account on databases supporting row locks.
            get_user_model().objects.select_for_update().get(pk=request.user.pk)
            if Memory.objects.filter(user=request.user).count() >= 20:
                raise ValidationError({'detail': 'You can save up to 20 memories. Remove one before adding another.'})
            serializer.save(user=request.user)
        return Response(serializer.data, status=201)

    def delete(self, request):
        Memory.objects.filter(user=request.user).delete()
        return Response(status=204)


class MemoryDetailView(AccountAPIView):
    def patch(self, request, pk):
        memory = get_object_or_404(Memory, pk=pk, user=request.user)
        serializer = MemorySerializer(memory, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)

    def delete(self, request, pk):
        get_object_or_404(Memory, pk=pk, user=request.user).delete()
        return Response(status=204)
