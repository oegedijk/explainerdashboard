
def add_header(html:str, title="explainerdashboard"):
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<title>{title}</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
</head>
<body>

<div class="container">
{html}
</div>

</body>
</html>
"""

def card(html, title=None, subtitle=None):
    return f"""
<div class="card">
  <div class="card-header">
    <h3 class="card-title">{title}</h5>
    <h6 class="card-subtitle">{subtitle}</h5>

  </div>
  <div class="card-body">
    <div>
    {html}
    </div>
  </div>
</div>
"""
    
def wrap_in_div(html:str):
    return f"<div>{html}</div>"


def table_from_df(df):
    header_row = '\n'.join([f'      <th scope="col">{col}</th>' for col in df.columns])
    body_rows = ""
    for i, row in df.iterrows():
        body_rows += ('    <tr>\n'+'\n'.join(["      <td>"+str(val)+"</td>" for val in row.values])+'\n    </tr>\n')
        
    table = f"""
<table class="table">
  <thead>
    <tr>
{header_row}
    </tr>
  </thead>
  <tbody>
{body_rows}
  </tbody>
</table>
    """
    return table