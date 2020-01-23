---
rplogo: Red Points Logo Recognition library
---

See project info in this [confluence
link](https://confluence.rdpnts.com/display/DS/Logo+extraction).

This library provides functionality to write code for logo recognition problems.

Installation
============

Easy install
------------

Install the package using ```easy_install```:

``` {.bash}
$ easy_install rplogo
```

You can also install from a local copy of the repository:

``` {.bash}
$ easy_install .
```

Pip
---

Install the package using [pip](https://pip.pypa.io/en/stable/):

``` {.bash}
$ pip install rplogo
```

You can also install from a local copy of the repository:

``` {.bash}
$ pip install .
```

Usage
=====

Import the library in your python code as usual:

    import rplogo
    from rplogo import ...

The rplogo provides a an interface to interact with LogoGrab service.

There are 2 main methods and a wrapper for both of them.

    send_ondemand

* Sends an image url to logograb service and gets the request_hash if the call went well. Otherwise raises an exception  

    read_ondemand

* Sends the request_hash to logograb to extract the results from the image_url.
  Raises an exception if theres no result. Can be configured the number of retries to send to logograb to get the information and the cadence using max_retries and backoff_factor.
    
    
Compatibility
=============

This package is tested against Python 3.6.

Licence
=======

RedPoints Proprietary License

Authors
=======

```rplogo``` was written by [Red
Points](dev@redpoints.com).