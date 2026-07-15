.. SlickTune documentation master file.
   Pattern mirrors slick-ml docs (https://github.com/slickml/slick-ml/tree/master/docs).

   NOTES:
   1) Create new pages under ``docs/pages/`` in ``.md`` or ``.rst`` format
   2) Add Sphinx extensions in ``docs/conf.py``
   3) ``docs/pages/releases.md`` is a symlink to ``CHANGELOG.md``
   4) ``docs/pages/contributing.md`` is a symlink to ``CONTRIBUTING.md``

SlickTune 🧩 Documentation
**************************

|build_status| |codecov| |downloads| |github_stars| |slack_invite| |twitter_url|

----

🧠 SlickTune 🧩 Philosophy
--------------------------

`SlickTune <https://github.com/slickml/slick-tune>`_ is SlickML's composable toolkit for
fine-tuning large language models. Fine-tuning is treated as an orthogonal stack —

.. code-block:: text

   model  ×  strategy  ×  objective  ×  data  ×  metrics

— so you can swap LoRA for DoRA / AdaLoRA / QLoRA / full FT without rewriting the rest.
Built on Transformers + PEFT + TRL, with probes and holdout metrics so you can verify the
model actually learned *your* facts.

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`🛠 Installation <pages/installation>`
        :link: pages/installation
        :link-type: doc

        Requirements and how to set up your Python environment with ``uv`` ...

    .. grid-item-card:: :doc:`📌 Quick Start <pages/quick_start>`
        :link: pages/quick_start
        :link-type: doc

        Train LoRA SFT, probe facts, and run holdout eval in a few commands ...

    .. grid-item-card:: :doc:`📘 Fine-Tuning Guide <pages/fine_tuning_guide>`
        :link: pages/fine_tuning_guide
        :link-type: doc

        Visual guide to Full FT, LoRA, DoRA, AdaLoRA, and QLoRA for beginners ...

    .. grid-item-card:: :doc:`🎯 API Reference <pages/api>`
        :link: pages/api
        :link-type: doc

        Explore the SlickTune API and source modules ...

    .. grid-item-card:: :doc:`📣 Changelog & Releases <pages/releases>`
        :link: pages/releases
        :link-type: doc

        Stay up-to-date with new features and fixes ...

----

🧑‍💻🤝 Become a Contributor
----------------------------
SlickTune is building an open-source community for practical LLM fine-tuning. Development
details live in our `Contributing <pages/contributing.html>`_ guidelines. Special thanks to
all contributors 👇

.. image:: https://contrib.rocks/image?repo=slickml/slick-tune
   :width: 100
   :alt: Contributors
   :target: https://github.com/slickml/slick-tune/graphs/contributors


.. image:: https://repobeats.axiom.co/api/embed/5205f02e274ac9df0d3b8fe80be684139ac0c878.svg
  :width: 1000
  :alt: Repobeats analytics image
  :target: https://github.com/slickml/slick-ml/commits/master
----

❓ 🆘 📲 Need Help?
----------------------
Please join our `Slack Channel <https://www.slickml.com/slack-invite>`_ to interact with the
core team and community, or email `admin@slickml.com <mailto:admin@slickml.com>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   Installation <pages/installation>
   Quick Start <pages/quick_start>
   Fine-Tuning Guide <pages/fine_tuning_guide>
   Releases <pages/releases>
   Contributing <pages/contributing>
   Citation <pages/citation>
   License <pages/license>
   Code of Conduct <pages/code_of_conduct>
   Contact Us <pages/contact_us>
   API Reference <pages/api>

----

🔍 Indices and Tables
-----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |build_status| image:: https://github.com/slickml/slick-tune/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/slickml/slick-tune/actions/workflows/ci.yml
.. |codecov| image:: https://codecov.io/gh/slickml/slick-tune/graph/badge.svg
   :target: https://codecov.io/gh/slickml/slick-tune
.. |downloads| image:: https://pepy.tech/badge/slicktune
   :target: https://pepy.tech/project/slicktune
.. |twitter_url| image:: https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2FSlickML
   :target: https://twitter.com/SlickML
.. |slack_invite| image:: https://badgen.net/badge/Join/SlickML%20Slack/purple?icon=slack
   :target: https://www.slickml.com/slack-invite
.. |github_stars| image:: https://img.shields.io/github/stars/slickml/slick-tune?color=cyan&label=github&logo=github
   :target: https://github.com/slickml/slick-tune
