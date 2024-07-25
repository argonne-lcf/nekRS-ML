"""
Definition and loading of standard environment variables, plus a wrappers for loading
and parsing values.

This does not include service URL env vars (see environments.py for loading of those)
"""

from __future__ import annotations

import logging
import os
import pathlib
import typing as t

log = logging.getLogger(__name__)
T = t.TypeVar("T")


ENVNAME_VAR = "GLOBUS_SDK_ENVIRONMENT"
HTTP_TIMEOUT_VAR = "GLOBUS_SDK_HTTP_TIMEOUT"
SSL_VERIFY_VAR = "GLOBUS_SDK_VERIFY_SSL"


@t.overload
def _load_var(
    varname: str,
    default: t.Any,
    explicit_value: t.Any | None,
    convert: t.Callable[[t.Any, t.Any], T],
) -> T: ...


@t.overload
def _load_var(
    varname: str,
    default: str,
    explicit_value: str | None,
) -> str: ...


def _load_var(
    varname: str,
    default: t.Any,
    explicit_value: t.Any | None = None,
    convert: t.Callable[[t.Any, t.Any], T] | None = None,
) -> t.Any:
    # use the explicit value if given and non-None, otherwise, do an env lookup
    value = (
        explicit_value if explicit_value is not None else os.getenv(varname, default)
    )
    if convert:
        value = convert(value, default)
    # only info log on non-default *values*
    # meaning that if we define the default as 'foo' and someone explicitly sets 'foo',
    # no info log gets emitted
    if value != default:
        log.info(f"on lookup, non-default setting: {varname}={value}")
    else:
        log.debug(f"on lookup, default setting: {varname}={value}")
    return value


def _ssl_verify_cast(
    value: t.Any, default: t.Any  # pylint: disable=unused-argument
) -> bool | str:
    if isinstance(value, bool):
        return value
    if not isinstance(value, (str, pathlib.Path)):
        msg = f"Value {value} of type {type(value)} cannot be used for SSL verification"
        raise ValueError(msg)
    if isinstance(value, str):
        if value.lower() in {"y", "yes", "t", "true", "on", "1"}:
            return True
        if value.lower() in {"n", "no", "f", "false", "off", "0"}:
            return False
        if os.path.isfile(value):
            return value
    if isinstance(value, pathlib.Path) and value.is_file():
        return str(value.absolute())
    raise ValueError(
        "SSL verification value must be a valid boolean value "
        f"or a path to a file that exists (got {value})"
    )


def _optfloat_cast(value: t.Any, default: t.Any) -> float | None:
    try:
        return float(value)
    except ValueError:
        pass
    if value == "":
        return t.cast(float, default)
    log.error(f'Value "{value}" can\'t cast to optfloat')
    raise ValueError(f"Invalid config float: {value}")


def get_environment_name(inputenv: str | None = None) -> str:
    return _load_var(ENVNAME_VAR, "production", explicit_value=inputenv)


def get_ssl_verify(value: bool | str | pathlib.Path | None = None) -> bool | str:
    return _load_var(
        SSL_VERIFY_VAR, default=True, explicit_value=value, convert=_ssl_verify_cast
    )


def get_http_timeout(value: float | None = None) -> float | None:
    ret = _load_var(
        HTTP_TIMEOUT_VAR, 60.0, explicit_value=value, convert=_optfloat_cast
    )
    if ret == -1.0:
        return None
    return ret
