% Nested sampling options

The type of bounding and sampling methods used are *crucial*.

The likelihood function with the new prior exhibits several sharp peaks within
each frequency.  Nested sampling performance is improved (and indeed actually
finishes runs) by:

- Using 'multi' as the bound method -- is the only viable alternative  

- Using random walk 'rslice' instead of uniform sampling. The order of
  preference of sampling  methods is: 'unif' < 'slice' ~ 'hslice' < 'rstagger'
  < 'rwalk' < 'rslice'
  
- Note that `rslice` is similar to Galilean Monte Carlo [@Speagle2019]

- Using bootstrapping to estimate expansion factor instead of the fixed
  default.  
       
  > Bootstrapping these expansion factors can help to ensure accurate evidence
  > estimation    when the proposals rely heavily on the size of an object
  > rather than the overall shape,  such as when proposing new points uniformly
  > within their boundaries. In theory, it also  helps to prevent mode "death":
  > if occasionally a secondary mode disappears when bootstrapping,  the
  > existing bounds would be expanded to theoretically encompass it. In
  > practice, however, most modes are widely separated, [as in our case]
  > leading enormous expansion factors  whenever any possible instance of mode
  > death may occur.[1]  
 
[1]: https://github.com/joshspeagle/dynesty/blob/master/demos/Examples%20--%20Gaussian%20Shells.ipynb 
