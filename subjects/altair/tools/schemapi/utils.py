"""Utilities for working with schemas"""

import keyword
import re
import textwrap
import urllib

import jsonschema


EXCLUDE_KEYS = ("definitions", "title", "description", "$schema", "id")


def resolve_references(schema, root=None):
    """Resolve References within a JSON schema"""
    resolver = jsonschema.RefResolver.from_schema(root or schema)
    while "$ref" in schema:
        with resolver.resolving(schema["$ref"]) as resolved:
            schema = resolved
    return schema


def get_valid_identifier(
    prop, replacement_character="", allow_unicode=False, url_decode=True
):
    """Given a string property, generate a valid Python identifier

    Parameters
    ----------
    prop: string
        Name of property to decode.
    replacement_character: string, default ''
        The character to replace invalid characters with.
    allow_unicode: boolean, default False
        If True, then allow Python 3-style unicode identifiers.
    url_decode: boolean, default True
        If True, decode URL characters in identifier names.

    Examples
    --------
    >>> get_valid_identifier('my-var')
    'myvar'

    >>> get_valid_identifier('if')
    'if_'

    >>> get_valid_identifier('$schema', '_')
    '_schema'

    >>> get_valid_identifier('$*#$')
    '_'

    >>> get_valid_identifier("Name%3Cstring%3E")
    'Namestring'
    """
    # Decode URL characters.
    if url_decode:
        prop = urllib.parse.unquote(prop)

    # Deal with []
    prop = prop.replace("[]", "Array")

    # First substitute-out all non-valid characters.
    flags = re.UNICODE if allow_unicode else re.ASCII
    valid = re.sub(r"\W", replacement_character, prop, flags=flags)

    # If nothing is left, use just an underscore
    if not valid:
        valid = "_"

    # first character must be a non-digit. Prefix with an underscore
    # if needed
    if re.match(r"^[\d\W]", valid):
        valid = "_" + valid

    # if the result is a reserved keyword, then add an underscore at the end
    if keyword.iskeyword(valid):
        valid += "_"
    return valid


def is_valid_identifier(var, allow_unicode=False):
    """Return true if var contains a valid Python identifier

    Parameters
    ----------
    val : string
        identifier to check
    allow_unicode : bool (default: False)
        if True, then allow Python 3 style unicode identifiers.
    """
    flags = re.UNICODE if allow_unicode else re.ASCII
    is_valid = re.match(r"^[^\d\W]\w*\Z", var, flags)
    return is_valid and not keyword.iskeyword(var)


class SchemaProperties:
    """A wrapper for properties within a schema"""

    def __init__(self, properties, schema, rootschema=None):
        self._properties = properties
        self._schema = schema
        self._rootschema = rootschema or schema

    def __bool__(self):
        return bool(self._properties)

    def __dir__(self):
        return list(self._properties.keys())

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return super(SchemaProperties, self).__getattr__(attr)

    def __getitem__(self, attr):
        dct = self._properties[attr]
        if "definitions" in self._schema and "definitions" not in dct:
            dct = dict(definitions=self._schema["definitions"], **dct)
        return SchemaInfo(dct, self._rootschema)

    def __iter__(self):
        return iter(self._properties)

    def items(self):
        return ((key, self[key]) for key in self)

    def keys(self):
        return self._properties.keys()

    def values(self):
        return (self[key] for key in self)


class SchemaInfo:
    """A wrapper for inspecting a JSON schema"""

    def __init__(self, schema, rootschema=None):
        if hasattr(schema, "_schema"):
            if hasattr(schema, "_rootschema"):
                schema, rootschema = schema._schema, schema._rootschema
            else:
                schema, rootschema = schema._schema, schema._schema
        elif not rootschema:
            rootschema = schema
        self.raw_schema = schema
        self.rootschema = rootschema
        self.schema = resolve_references(schema, rootschema)

    def child(self, schema):
        return self.__class__(schema, rootschema=self.rootschema)

    def __repr__(self):
        keys = []
        for key in sorted(self.schema.keys()):
            val = self.schema[key]
            rval = repr(val).replace("\n", "")
            if len(rval) > 30:
                rval = rval[:30] + "..."
            if key == "definitions":
                rval = "{...}"
            elif key == "properties":
                rval = "{\n    " + "\n    ".join(sorted(map(repr, val))) + "\n  }"
            keys.append('"{}": {}'.format(key, rval))
        return "SchemaInfo({\n  " + "\n  ".join(keys) + "\n})"

    @property
    def title(self):
        if self.is_reference():
            return get_valid_identifier(self.refname)
        else:
            return ""

    @property
    def short_description(self):
        if self.title:
            # use RST syntax for generated sphinx docs
            return ":class:`{}`".format(self.title)
        else:
            return self.medium_description

    _simple_types = {
        "string": "string",
        "number": "float",
        "integer": "integer",
        "object": "mapping",
        "boolean": "boolean",
        "array": "list",
        "null": "None",
    }

    @property
    def medium_description(self):
        if self.is_list():
            return "[{0}]".format(
                ", ".join(self.child(s).short_description for s in self.schema)
            )
        elif self.is_empty():
            return "Any"
        elif self.is_enum():
            return "enum({})".format(", ".join(map(repr, self.enum)))
        elif self.is_anyOf():
            return "anyOf({})".format(
                ", ".join(s.short_description for s in self.anyOf)
            )
        elif self.is_oneOf():
            return "oneOf({})".format(
                ", ".join(s.short_description for s in self.oneOf)
            )
        elif self.is_allOf():
            return "allOf({})".format(
                ", ".join(s.short_description for s in self.allOf)
            )
        elif self.is_not():
            return "not {}".format(self.not_.short_description)
        elif isinstance(self.type, list):
            options = []
            subschema = SchemaInfo(dict(**self.schema))
            for typ_ in self.type:
                subschema.schema["type"] = typ_
                options.append(subschema.short_description)
            return "anyOf({})".format(", ".join(options))
        elif self.is_object():
            return "Mapping(required=[{}])".format(", ".join(self.required))
        elif self.is_array():
            return "List({})".format(self.child(self.items).short_description)
        elif self.type in self._simple_types:
            return self._simple_types[self.type]
        elif not self.type:
            import warnings

            warnings.warn("no short_description for schema\n{}" "".format(self.schema))
            return "any"

    @property
    def long_description(self):
        # TODO
        return "Long description including arguments and their types"

    @property
    def properties(self):
        return SchemaProperties(
            self.schema.get("properties", {}), self.schema, self.rootschema
        )

    @property
    def definitions(self):
        return SchemaProperties(
            self.schema.get("definitions", {}), self.schema, self.rootschema
        )

    @property
    def required(self):
        return self.schema.get("required", [])

    @property
    def patternProperties(self):
        return self.schema.get("patternProperties", {})

    @property
    def additionalProperties(self):
        return self.schema.get("additionalProperties", True)

    @property
    def type(self):
        return self.schema.get("type", None)

    @property
    def anyOf(self):
        return [self.child(s) for s in self.schema.get("anyOf", [])]

    @property
    def oneOf(self):
        return [self.child(s) for s in self.schema.get("oneOf", [])]

    @property
    def allOf(self):
        return [self.child(s) for s in self.schema.get("allOf", [])]

    @property
    def not_(self):
        return self.child(self.schema.get("not", {}))

    @property
    def items(self):
        return self.schema.get("items", {})

    @property
    def enum(self):
        return self.schema.get("enum", [])

    @property
    def refname(self):
        return self.raw_schema.get("$ref", "#/").split("/")[-1]

    @property
    def ref(self):
        return self.raw_schema.get("$ref", None)

    @property
    def description(self):
        return self._get_description(include_sublevels=False)

    @property
    def deep_description(self):
        return self._get_description(include_sublevels=True)

    def _get_description(self, include_sublevels: bool = False):
        desc = self.raw_schema.get("description", self.schema.get("description", ""))
        if not desc and include_sublevels:
            for item in self.anyOf:
                sub_desc = item._get_description(include_sublevels=False)
                if desc and sub_desc:
                    raise ValueError(
                        "There are multiple potential descriptions which could"
                        + " be used for the currently inspected schema. You'll need to"
                        + " clarify which one is the correct one.\n"
                        + str(self.schema)
                    )
                if sub_desc:
                    desc = sub_desc
        return desc

    def is_list(self):
        return isinstance(self.schema, list)

    def is_reference(self):
        return "$ref" in self.raw_schema

    def is_enum(self):
        return "enum" in self.schema

    def is_empty(self):
        return not (set(self.schema.keys()) - set(EXCLUDE_KEYS))

    def is_compound(self):
        return any(key in self.schema for key in ["anyOf", "allOf", "oneOf"])

    def is_anyOf(self):
        return "anyOf" in self.schema

    def is_allOf(self):
        return "allOf" in self.schema

    def is_oneOf(self):
        return "oneOf" in self.schema

    def is_not(self):
        return "not" in self.schema

    def is_object(self):
        if self.type == "object":
            return True
        elif self.type is not None:
            return False
        elif (
            self.properties
            or self.required
            or self.patternProperties
            or self.additionalProperties
        ):
            return True
        else:
            raise ValueError("Unclear whether schema.is_object() is True")

    def is_value(self):
        return not self.is_object()

    def is_array(self):
        return self.type == "array"

    def schema_type(self):
        if self.is_empty():
            return "empty"
        elif self.is_compound():
            for key in ["anyOf", "oneOf", "allOf"]:
                if key in self.schema:
                    return key
        elif self.is_object():
            return "object"
        elif self.is_array():
            return "array"
        elif self.is_value():
            return "value"
        else:
            raise ValueError("Unknown type with keys {}".format(self.schema))

    def property_name_map(self):
        """
        Return a mapping of schema property names to valid Python attribute names

        Only properties which are not valid Python identifiers will be included in
        the dictionary.
        """
        pairs = [(prop, get_valid_identifier(prop)) for prop in self.properties]
        return {prop: val for prop, val in pairs if prop != val}


def indent_arglist(args, indent_level, width=100, lstrip=True):
    """Indent an argument list for use in generated code"""
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=indent_level * " ",
        subsequent_indent=indent_level * " ",
        break_long_words=False,
    )
    wrapped = "\n".join(wrapper.wrap(", ".join(args)))
    if lstrip:
        wrapped = wrapped.lstrip()
    return wrapped


def indent_docstring(lines, indent_level, width=100, lstrip=True):
    """Indent a docstring for use in generated code"""
    final_lines = []

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped:
            leading_space = len(line) - len(stripped)
            indent = indent_level + leading_space
            wrapper = textwrap.TextWrapper(
                width=width - indent,
                initial_indent=indent * " ",
                subsequent_indent=indent * " ",
                break_long_words=False,
                break_on_hyphens=False,
                drop_whitespace=True,
            )
            list_wrapper = textwrap.TextWrapper(
                width=width - indent,
                initial_indent=indent * " " + "* ",
                subsequent_indent=indent * " " + "  ",
                break_long_words=False,
                break_on_hyphens=False,
                drop_whitespace=True,
            )
            for line in stripped.split("\n"):
                line_stripped = line.lstrip()
                line_stripped = fix_docstring_issues(line_stripped)
                if line_stripped == "":
                    final_lines.append("")
                elif line_stripped.startswith("* "):
                    final_lines.extend(list_wrapper.wrap(line_stripped[2:]))
                # Matches lines where an attribute is mentioned followed by the accepted
                # types (lines starting with a character sequence that
                # does not contain white spaces or '*' followed by ' : ').
                # It therefore matches 'condition : anyOf(...' but not '**Notes** : ...'
                # These lines should not be wrapped at all but appear on one line
                elif re.match(r"[^\s*]+ : ", line_stripped):
                    final_lines.append(indent * " " + line_stripped)
                else:
                    final_lines.extend(wrapper.wrap(line_stripped))

        # If this is the last line, put in an indent
        elif i + 1 == len(lines):
            final_lines.append(indent_level * " ")
        # If it's not the last line, this is a blank line that should not indent.
        else:
            final_lines.append("")
    # Remove any trailing whitespaces on the right side
    stripped_lines = []
    for i, line in enumerate(final_lines):
        if i + 1 == len(final_lines):
            stripped_lines.append(line)
        else:
            stripped_lines.append(line.rstrip())
    # Join it all together
    wrapped = "\n".join(stripped_lines)
    if lstrip:
        wrapped = wrapped.lstrip()
    return wrapped


def fix_docstring_issues(docstring):
    # All lists should start with '*' followed by a whitespace. Fixes the ones
    # which either do not have a whitespace or/and start with '-' by first replacing
    # "-" with "*" and then adding a whitespace where necessary
    docstring = re.sub(
        r"^-(?=[ `\"a-z])",
        "*",
        docstring,
        flags=re.MULTILINE,
    )
    # Now add a whitespace where an asterisk is followed by one of the characters
    # in the square brackets of the regex pattern
    docstring = re.sub(
        r"^\*(?=[`\"a-z])",
        "* ",
        docstring,
        flags=re.MULTILINE,
    )

    # Links to the vega-lite documentation cannot be relative but instead need to
    # contain the full URL.
    docstring = docstring.replace(
        "types#datetime", "https://vega.github.io/vega-lite/docs/datetime.html"
    )
    return docstring
