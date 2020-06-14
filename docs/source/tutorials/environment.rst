.. _`env_tute`:

Interacting with NASim Environment
==================================

Assuming you understand scenarios and are comfortable loading an environment from a scenario (see :ref:`scenarios_tute` if not), then interacting with a NASim Environment is very easy and follows the same interface as `OpenAI gym <https://github.com/openai/gym>`_::

  import nasim
  # load my environment in the desired way (make_benchmark, load, generate)
  env = nasim.make_benchmark("tiny")
