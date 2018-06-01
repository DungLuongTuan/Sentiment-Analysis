$(document).ready(function() {
	let $input = $('#input');
	let $processing = $('.processing')
	let $result = $('.result')
	var $send_button = $('#send_button');

	function predictSentiment() {
		let text = $input.val();
		console.log(text)
		$result.css('visibility', 'hidden');
		$processing.css('display', 'block');
		$.ajax({
			url: '/',
			data: {'req': text},
			type: 'POST',
			success: function(response) {
				$processing.css('display', 'none')
				var res = JSON.parse(response)
				console.log(Object.keys(res))
				for(var key in res){
					console.log(key)
					var id = $('#' + key)
					var $exp_name = $(id[0].getElementsByClassName("exp_name")[0])
					var $positive = $(id[0].getElementsByClassName("positive")[0])
					var $negative = $(id[0].getElementsByClassName("negative")[0])

					$positive.css('width', (60*res[key]['positive']+4).toString() + '%')
					$positive.html((Math.round(res[key]['positive']*10000)/100).toString() + '%')
					$negative.css('width', (60*res[key]['negative']+4).toString() + '%')
					$negative.html((Math.round(res[key]['negative']*10000)/100).toString() + '%')
					id.css('visibility', 'visible')
				}
			},
			error: function(error) {
				console.log(error)
			}
		});
	}

	$send_button.click(function(){
		predictSentiment();
	})
})