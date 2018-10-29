# Contributing to PennyLane-SF

Thank you for taking the time to contribute to the PennyLane plugin ecosystem!
:confetti_ball: :tada: :fireworks: :balloon:

PennyLane and its plugins are a collaborative effort with the quantum computation and machine learning
community - while we will continue working on adding new and exciting features to PennyLane,
we invite you to join us and suggest improvements, research ideas, or even just to discuss how
PennyLane fits into your workflow.

## How can I get involved in the community?

If you want to contribute but don't know where to start, start by checking out the main
[PennyLane documentation](https://pennylane.readthedocs.io). Have a go working through some of the tutorials,
and having a look at the PennyLane API and code documentation to see how things work under the hood.

Finally, check out the contributing guidelines in the main
[PennyLane repository](https://github.com/XanaduAI/openqml/blob/master/.github/CONTRIBUTING.md).

If you are interested in contributing to the PennyLane-SF plugin directly, continue reading below.

## How can I contribute to PennyLane-SF?

* **Report bugs** - even if you are not using the development branch of PennyLane-SF, if you come
  across any bugs or issues, make a bug report. See the next section for more details on the bug
  reporting procedure.

* **Suggest new features and enhancements** - use the GitHub issue tracker
  and let us know what will make PennyLane-SF even better for your workflow.

* **Contribute to our documentation, or to PennyLane-SF directly** - if you would like to add
  to our documentation, or suggest improvements/changes, let us know - or even submit a pull request directly. All authors
  who have contributed to the PennyLane codebase will be listed alongside the latest release.

Appetite whetted? Keep reading below for all the nitty-gritty on reporting bugs, contributing to the documentation,
and submitting pull requests.

## Reporting bugs

We use the [GitHub issue tracker](https://github.com/XanaduAI/pennylane-sf/issues) to keep track of all reported
bugs and issues. If you find a bug, or have an issue with PennyLane-SF, please submit a bug report! User
reports help us make PennyLane better on all fronts.

To submit a bug report, please work your way through the following checklist:

* **Search the issue tracker to make sure the bug wasn't previously reported**. If it was, you can add a comment
  to expand on the issue if you would like.

* **Fill out the issue template**. If you cannot find an existing issue addressing the problem, create a new
  issue by filling out the [issue template](.github/ISSUE_TEMPLATE.md). This template is added automatically to the comment
  box when you create a new issue. Please try and add as many details as possible!

* Try and make your issue as **clear, concise, and descriptive** as possible. Include a clear and descriptive title,
  and include all code snippets/commands required to reproduce the problem. If you're not sure what caused the issue,
  describe what you were doing when the issue occurred.

## Suggesting features, document additions, and enhancements

To suggest features and enhancements, please use the GitHub tracker. There is no template required for
feature requests and enhancements, but here are a couple of suggestions for things to include.

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested feature**.

  - If the feature is related to any theoretical results in quantum machine learning or quantum computation,
    provide any relevant equations. Alternatively, provide references to papers/preprints,
    with the relevant sections/equations noted.
  - If the feature is workflow-related, or related to the use of PennyLane,
    explain why the enhancement would be useful, and where/how you would like to use it.

* **For documentation additions**, point us towards any relevant equations/papers/preprints,
    with the relevant sections/equations noted. Short descriptions of its use/importance would also be useful.

## Pull requests

If you would like to contribute directly to the PennyLane-SF codebase, simply make a fork of the master branch, and
then when ready, submit a [pull request](https://help.github.com/articles/about-pull-requests). We encourage everyone
to have a go forking and modifying the PennyLane source code, however, we have a couple of guidelines on pull
requests to ensure the main master branch of PennyLane conforms to existing standards and quality.

### General guidelines

* **Do not make a pull request for minor typos/cosmetic code changes** - make an issue instead.
* **For major features, consider making an independent app** that runs on top of PennyLane, rather than modifying
  PennyLane directly.

### Before submitting

Before submitting a pull request, please make sure the following is done:

* **All new features must include a unit test.** If you've fixed a bug or added code that should be tested,
  add a test to the test directory!
* **All new functions and code must be clearly commented and documented.** Have a look through the source code at some of
  the existing function docstrings - the easiest approach is to simply copy an existing docstring and modify it as appropriate.
  If you do make documentation changes, make sure that the docs build and render correctly by running `make docs`.
* **Ensure that the test suite passes**, by running `make test`.
* **Make sure the modified code in the pull request conforms to the PEP8 coding standard.** The PennyLane-SF source code
  conforms to [PEP8 standards](https://www.python.org/dev/peps/pep-0008/). We check all of our code against
  [Pylint](https://www.pylint.org/). To lint modified files, simply install `pip install pylint`, and then from the source code
  directory, run `pylint pennylane/path/to/file.py`.

### Submitting the pull request
* When ready, submit your fork as a [pull request](https://help.github.com/articles/about-pull-requests) to the PennyLane
  repository, filling out the [pull request template](.github/PULL_REQUEST_TEMPLATE.md). This template is added automatically
  to the comment box when you create a new issue.

* When describing the pull request, please include as much detail as possible regarding the changes made/new features
  added/performance improvements. If including any bug fixes, mention the issue numbers associated with the bugs.

* Once you have submitted the pull request, three things will automatically occur:

  - The **test suite** will automatically run on [Travis CI](https://travis-ci.org/XanaduAI/pennylane)
    to ensure that the all tests continue to pass.
  - Once the test suite is finished, a **code coverage report** will be generated on
    [Codecov](https://codecov.io/gh/XanaduAI/pennylane). This will calculate the percentage of PennyLane
    covered by the test suite, to ensure that all new code additions are adequately tested.
  - Finally, the **code quality** is calculated by [Codacy](https://app.codacy.com/app/XanaduAI/pennylane/dashboard),
    to ensure all new code additions adhere to our code quality standards.

  Based on these reports, we may ask you to make small changes to your branch before merging the pull request into the master branch. Alternatively, you can also
  [grant us permission to make changes to your pull request branch](https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/).

:fireworks: Thank you for contributing to PennyLane! :fireworks:

\- The PennyLane team
