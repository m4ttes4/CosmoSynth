### 🚧 **Work in Progress!** 🚧

> ⚠️ **CosmoSynth is still a work in progress!**  
> It’s not mature or intended to be _production-ready_. Expect significant changes—many of which are happening outside this repo!

## 🚀 **Why CosmoSynth?**

CosmoSynth is more than just a library—it's an idea: making it easy and _Pythonic_ to build flexible yet powerful models.

There are already well-established, battle-tested libraries for modeling and fitting, such as **Astropy** and **LmFit**, both of which greatly inspire CosmoSynth. However, they have some limitations:

- **Astropy** doesn’t natively support fitting custom models with MCMC libraries like `emcee`.
- **LmFit** does, but it often introduces computational overhead, sacrificing performance.

CosmoSynth aims to **bridge the gap**, combining the best of both worlds: an intuitive interface **without** compromising efficiency. 

### ✅ **What You Can Do**
Here’s what **CosmoSynth** supports right now:
- 🔧 **Define models** from user-defined functions.
- 🔗 **Combine models** using various operations.
- 🎯 **Add or remove constraints** of different types on individual model parameters.
- 🔒 **Fix parameters** when needed.
- 📊 **Fit models to observational data** with a direct interface and multiple optimization methods.