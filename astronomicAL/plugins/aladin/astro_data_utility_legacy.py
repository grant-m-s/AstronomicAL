from string import Template


def make_srcdoc_aladin_lite(survey_id, ra, dec, fov = 0.08):
    tpl = Template("""<!doctype html>
            <html><head>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, height=device-height, initial-scale=1.0, user-scalable=no">
            <style>html,body,#aladin{margin:0;width:100%;height:100%}</style>
            </head><body>
            <div id="aladin"></div>
            <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js" charset="utf-8"></script>
            <script>
            A.init.then(function () {
                A.aladin("#aladin", {
                cooFrame: "ICRSd",
                survey: "$survey",
                target: "$ra $dec",
                fov: $fov,
                showFullscreenControl: false,
                showLayersControl: false
                });
            });
            </script>
            </body></html>""")
    return tpl.substitute(survey=survey_id, ra = ra, dec =dec, fov=fov)