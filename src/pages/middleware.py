from django.utils.deprecation import MiddlewareMixin

class CustomSessionMiddleware(MiddlewareMixin):
    def process_request(self, request):
        # Example: Set session key prefix based on URL path
        if request.path.startswith('/'):
            request.session._session_key_prefix = 'view1'
        elif request.path.startswith('/practice1/'):
            request.session._session_key_prefix = 'view2'