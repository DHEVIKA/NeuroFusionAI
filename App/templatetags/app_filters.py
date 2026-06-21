from django import template

register = template.Library()

@register.filter(name='replace')
def replace(value, arg):
    """
    Replaces occurrences of a substring with another substring in a template.
    E.g., {{ value|replace:"_, " }} will replace "_" with " ".
    """
    if not isinstance(value, str):
        return value
    
    parts = arg.split(',')
    if len(parts) == 2:
        return value.replace(parts[0], parts[1])
    return value.replace(arg, ' ')
