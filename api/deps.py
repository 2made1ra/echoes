"""Зависимости слоя API: доступ к сервису, собранному при старте приложения."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from services.dialogue import DialogueService


def get_service(request: Request) -> DialogueService:
    return request.app.state.service


ServiceDep = Annotated[DialogueService, Depends(get_service)]
