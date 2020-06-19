# Tensorboard2Report
> Painless report generation from Tensorboard logs.

<!-- [![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url] -->

Getting from a lots of Tensorboard logs to a report ready table/plot is somewhat tedious, particularly if you want to do an hyperparameter sweep and report/plot it later.
This small project aims to alleviate this issue:
- specify the variables you want to have in your report
- specify how you want to reduce the experiment to a single datapoint : latest entry ? maximum ?
- get a simple .csv and plug it in your prefered tool to generate reports (Latex/PGFplot oriented .csv organisation)

<!-- ![](header.png) -->

## Installation and usage

Pull the project, setup your `filter.yaml` file. Then install the required packages and run like this:
```sh
pip install -r requirements.txt
python3 main.py --filters=filter.yaml --metrics Cons/act_cons
```
<!-- OS X & Linux:

```sh
npm install my-crazy-module --save
```

Windows:

```sh
edit autoexec.bat
``` -->

## Parameters
If you would like to use the script of the shelf, here are the possible parameters you can pass.

- `--filters`: the path to your `filter.yaml` file. See it as a simple dictionnary of Filters types and parameters.
- `--logdir`: the path to your Tensorboard logs. The structure should look like :
    - `experiment_xxx``
        - `hparams.yaml` (hyperparameters for this experiment)
        - `events.[...]` (Tensorboard generated files)
    - `experiment_yyy``
        - `hparams.yaml` (hyperparameters for this experiment)
        - `events.[...]` (Tensorboard generated files)
- `--groupby`: for which hyperparameters would you like your experiment to be grouped ?
- `--target`: based on which metric should the report be made ?
- `--reduction`: how will the whole experiment be reduced to one datapoint ? That is controlled by an enum, so you can either pick `Reduction.max` or `Reduction.last`.
    - E.g. if `target=top1_accuracy`, if `reduction=Reduction.max` the experiments will be reduced to the maximal `top1_accuracy` entry.
- `--metrics`: the variables which you want to report. Use like `--metrics Depth Width` if you want to report on the variables `Depth` and `Width`

<!-- _For more examples and usage, please refer to the [Wiki][wiki]._ -->

## Contributing
Feel free to suggest improvements/pull requests c:

## Release History

* 0.0.1 (19/06/20)
    *  Initial release
    * Support for atomic filters

## Meta

<!-- Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com -->

Distributed under the MIT license. See ``LICENSE`` for more information.

<!-- [https://github.com/yourname/github-link](https://github.com/dbader/) -->
<!-- 
## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request -->

<!-- Markdown link & img dfn's -->
<!-- [npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki -->