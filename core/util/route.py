from copy import copy

from fastapi import APIRouter, FastAPI


def include_api_router(app: FastAPI, router: APIRouter, prefix: str) -> None:
    """CBV의 '/' 루트를 '/api/chat'처럼 슬래시 없는 그룹 경로에 등록합니다.

    fastapi-utils는 내부 include_router 때문에 빈 경로를 허용하지 않습니다.
    등록용 복사본의 경로만 조정하고 FastAPI가 최종 라우트를 다시 구성하게 합니다.
    """
    mounted = copy(router)
    mounted.routes = [copy(route) for route in router.routes]
    for route in mounted.routes:
        if getattr(route, "path", None) == "/":
            route.path = ""
        elif hasattr(route, "original_router"):
            # FastAPI 0.141의 지연 include_router: CBV 하위 라우터를 복사합니다.
            child = copy(route.original_router)
            child.routes = [copy(item) for item in child.routes]
            for item in child.routes:
                if getattr(item, "path", None) == "/":
                    item.path = ""
            route.original_router = child
    app.include_router(mounted, prefix=prefix)
