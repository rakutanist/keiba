/** Download・処理 */
let downloadCSV = (data, name) => {
  /** Blob Object を作成する Type. CSV */
  const blob = new Blob([data], { type: "text/csv" });
  console.log("blob", blob);
  const url = window.URL.createObjectURL(blob);
  console.log("url", url);
  const a = document.createElement("a");
  a.setAttribute("href", url);
  a.setAttribute("download", `${name}.csv`);
  a.click();
  a.remove();
};

let filename = document.querySelector('#contents');
let element = document.getElementsByTagName('tr');
let num = element.length;
console.log(num);
let arr = Array(num - 1);
for (let i = 1, len = num; i < len; i++){ arr[i-1] = element[i].querySelector('.race_num').getElementsByTagName('a')[0].attributes[0].nodeValue };
str = arr.join(',');
let id = filename.querySelector('.opt').innerHTML.replace(' ', '');
console.log(str);
console.log(filename)

downloadCSV(arr, `URL_child_${id}`);