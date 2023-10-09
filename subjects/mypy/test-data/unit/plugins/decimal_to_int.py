from mypy.plugin import Plugin


class MyPlugin(Plugin):
    def get_type_analyze_hook(self, fullname):
        if fullname in ("decimal.Decimal", "_decimal.Decimal"):
            return decimal_to_int_hook
        return None

def plugin(version):
    return MyPlugin

def decimal_to_int_hook(ctx):
    return ctx.api.named_type('builtins.int', [])
