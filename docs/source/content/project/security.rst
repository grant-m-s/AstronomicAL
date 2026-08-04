Security Policy
===============

Reporting a Security Issue
--------------------------

Do not publish credentials, private dataset paths or a working exploit in a
public issue.

.. todo::

   Add the private security-reporting address or repository security advisory
   process.

Plugin Security
---------------

Plugins execute local Python code. Review third-party plugins before enabling
them.

Credentials
-----------

Tokens and passwords should be supplied through the environment or a dedicated
credential mechanism. They should not be saved in layouts.

Remote Content
--------------

Treat remote dataset code, model files and executable checkpoints as untrusted
unless their source has been reviewed.

Sensitive Data
--------------

A layout may contain dataset paths, row identifiers and workflow settings.
Review it before sharing.
